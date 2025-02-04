# bot.py
from collections import deque
import cityhash
import discord
import os
import json
import logging
import re
from report import Report, State
import pdb
# classifier requirements
import io
import requests
import numpy as np
from PIL import Image
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from classifier.classifier import NN


# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']


class ModBot(discord.Client):
    REPORTEE_THRESHOLD = 0
    REPORTER_THRESHOLD = 0
    REPORTER_THRESHOLD_PERCENTAGE = 0.0

    def __init__(self): 
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='.', intents=intents)

        self.group_num = None
        self.mod_channels = {} # Map from guild to the mod channel id for that guild
        # self.reports = {} # Map from user IDs to the state of their report
        self.reports = {
            'user_csam': deque([]),
            'user_adult': deque([]),
            'bot_csam': deque([]),
            'bot_adult': deque([])
        }
        self.queue_num = ['user_csam', 'user_adult', 'bot_csam', 'bot_adult']
        self.active_reporters = {}
        self.active_moderators = {}
        self.curr_report = None # Sets the Report currently being handled by moderator
        self.curr_report_idx = None
        self.warned_users = set()  # Set of users who have been warned for adult nudity
        self.report_table = dict([("sample.comment.url", dict([(cityhash.CityHash128("this is a sample comment"), "dummy-result")]))])
        self.reporter_table = {}
        self.reportee_table = {}
        self.targetted_users = {}
        # neural network model for automated image classification
        self.model = NN.load_from_checkpoint("classifier/lightning_logs/version_0/checkpoints/epoch=30-step=43617.ckpt")
        self.model.eval()

    def print_tables(self):
        print("number of active reporters")
        print(len(self.active_reporters))
        print("number of reports in report queues")
        print(self.reports)
        print("Reporter table")
        print(self.reporter_table)
        print("Reportee table")
        print(self.reportee_table)
        print("Report table")
        print(self.report_table)

    def print_targetted_users_table(self):
        for user, report_list in self.targetted_users:
            print(user + ": ")
            for i in range(1, len(report_list) + 1):
                print(f"{i}/ {report_list[i - 1]}")

    # Given a report, adds it to the report queue for reports made by bots.
    def add_report_to_bot_queue(self, report):
        if self.in_report_table(report):
            self.update_tables(report)
            return

        if report.report_adult():
            self.reports['bot_adult'].append(report)
        elif report.report_csam():
            self.reports['bot_csam'].append(report)
        return

    # Given a link to a message and a valid classification, makes a report for that message and adds it to the report
    # queue meant for bots.
    #
    # message_link: discord message link
    # classification: either Report.ADULT or Report.CSAM
    async def add_message_to_bot_queue(self, message, classification):
        report = Report(self)
        report.reporter = "bot"
        report.state = State.AWAITING_MESSAGE
        await report.handle_message(message, bot=True)
        report.state = classification
        self.add_report_to_bot_queue(report)
        return

    def in_report_table(self, report):
        if self.report_table.keys().__contains__(report.link):
            if self.report_table[report.link].keys().__contains__(cityhash.CityHash128(report.message.content)):
                return True
        return False

    # won't overwrite an existing entry
    def add_to_report_table(self, report, result):
        if self.report_table.keys().__contains__(report.link):
            if self.report_table[report.link].keys().__contains__(cityhash.CityHash128(report.message.content)):
                return
            self.report_table[report.link][cityhash.CityHash128(report.message.content)] = result
        else:
            self.report_table[report.link] = dict([(cityhash.CityHash128(report.message.content), result)])

    # returns the result on success, returns None on failure
    def report_table_result(self, report):
        if not self.in_report_table(report):
            return None
        return self.report_table[report.link][cityhash.CityHash128(report.message.content)]

    # reporter table structure: {string user : [int good, int bad]}, ex: {emilyc02 : [1, 0], stilakid: [3, 1]}
    # good/bad are 1 if adding to good/bad report count, 0 if not
    def update_reporter(self, user, good, bad):
        if user in self.reporter_table:
            self.reporter_table.update({user : [self.reporter_table[user][0] + good, self.reporter_table[user][1] + bad]})
        else:
            self.reporter_table[user] = [good, bad]

    def in_reporter_table(self, user):
        return user in self.reporter_table

    def reporter_is_up_to_no_good(self, user):
        if user in self.reporter_table and\
                self.reporter_table[user][0] + self.reporter_table[user][1] >= self.REPORTER_THRESHOLD and\
                self.in_reporter_table(user) and float(self.reporter_table[user][0]) / float(self.reporter_table[user][1]) >= self.REPORTER_THRESHOLD_PERCENTAGE:
            return True
        return False
    
    # reporter table structure: {string user : [{dict comment: int reports}, int warned]}
    # warned is 1 if adding warning, 0 if not
    def update_reportee(self, user, comment_url, warned):
        if user in self.reportee_table:
            if warned:
                self.reportee_table[user][1] += warned
            if comment_url:
                self.reportee_table[user][0][comment_url] = self.reportee_table[user][0].get(comment_url, 0) + 1
        else:
            self.reportee_table[user] = [{comment_url: 1}, warned]

    def in_reportee_table(self, user):
        return user in self.reportee_table

    def reportee_is_targetted(self, user, comment_url):
        if self.in_reportee_table(user) and self.reportee_table[user][0][comment_url] >= self.REPORTEE_THRESHOLD:
            return True
        return False

    # update table if report is already in the report table
    def update_tables(self, report):
        result = self.report_table_result(report)
        if result == "valid" or result == "wrong-type":
            # update reporter table
            self.update_reporter(report.reporter, 1, 0)
        elif result == "invalid":
            # update reporter table
            self.update_reporter(report.reporter, 0, 1)
            # update reportee table
            self.update_reportee(report.reportedUser, report.link, 1)

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is these guilds:')
        for guild in self.guilds:
            print(f' - {guild.name}')
        print('Press Ctrl-C to quit.')

        # Parse the group number out of the bot's name
        match = re.search('[gG]roup (\d+) [bB]ot', self.user.name)
        if match:
            self.group_num = match.group(1)
        else:
            raise Exception("Group number not found in bot's name. Name format should be \"Group # Bot\".")

        # Find the mod channel in each guild that this bot should report to
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == f'group-{self.group_num}-mod':
                    self.mod_channels[guild.id] = channel
                    self.mod_channel = channel
        
    async def on_message(self, message):
        '''
        This function is called whenever a message is sent in a channel that the bot can see (including DMs). 
        Currently the bot is configured to only handle messages that are sent over DMs or in your group's "group-#" channel. 
        '''
        # Ignore messages from the bot 
        if message.author.id == self.user.id:
            return

        # Check if this message was sent in a server ("guild") or if it's a DM
        if message.guild:
            await self.handle_channel_message(message)
        else:
            await self.handle_dm(message)
        self.print_tables()

    async def handle_dm(self, message):
        # Handle a help message
        if message.content == Report.HELP_KEYWORD:
            reply =  "Use the `report` command to begin the reporting process.\n"
            reply += "Use the `cancel` command to cancel the report process.\n"
            await message.channel.send(reply)
            return
        else:
            responses = []
            # Only respond to messages if they're part of a reporting flow
            # if not message.content.startswith(Report.START_KEYWORD):
            #     return

            if message.content.startswith(Report.START_KEYWORD):
                self.active_reporters[message.author.id] = Report(self)

            # Let the report class handle this message and generate a response.
            # Then, this function sends the response to the user.
            if message.author.id in self.active_reporters:
                responses = await self.active_reporters[message.author.id].handle_message(message)
                for r in responses:
                    await message.channel.send(r)

                # If the report needs to be moderated, it puts it in the correct queue and removes it from the list of acitve reports
                if self.active_reporters[message.author.id].report_csam():
                    if self.reporter_is_up_to_no_good(message.author.name):
                        await self.mod_channel.send(f"Report created by user flagged for invalid reports. Placed in hidden queue.")
                        return
                    elif self.in_report_table(self.active_reporters[message.author.id]):
                        self.update_tables(self.active_reporters[message.author.id])
                        if self.reportee_is_targetted(self.active_reporters[message.author.id].reportedUser, self.active_reporters[message.author.id].link):
                            if self.active_reporters[message.author.id].reportedUser not in self.targetted_users:
                                self.targetted_users[self.active_reporters[message.author.id].reportedUser] = []
                            self.targetted_users[self.active_reporters[message.author.id].reportedUser].append(self.active_reporters[message.author.id].link)
                    else:
                        self.reports['user_csam'].append(self.active_reporters[message.author.id])
                    self.active_reporters.pop(message.author.id)
                    await self.mod_channel.send(f"Report created by user - type \"queue\" to view outstanding reports.")

                elif self.active_reporters[message.author.id].report_adult():
                    if self.reporter_is_up_to_no_good(message.author.name):
                        await self.mod_channel.send(f"Report created by user flagged for invalid reports. Placed in hidden queue.")
                        return
                    elif self.in_report_table(self.active_reporters[message.author.id]):
                        self.update_tables(self.active_reporters[message.author.id])
                        if self.reportee_is_targetted(self.active_reporters[message.author.id].reportedUser, self.active_reporters[message.author.id].link):
                            if self.active_reporters[message.author.id].reportedUser not in self.targetted_users:
                                self.targetted_users[self.active_reporters[message.author.id].reportedUser] = []
                            self.targetted_users[self.active_reporters[message.author.id].reportedUser].append(self.active_reporters[message.author.id].link)
                    else:
                        self.reports['user_adult'].append(self.active_reporters[message.author.id])
                    self.active_reporters.pop(message.author.id)
                    await self.mod_channel.send(f"Report created by user - type \"queue\" to view outstanding reports.")

                # If the report is complete or cancelled, remove it from our map
                elif self.active_reporters[message.author.id].report_complete():
                    self.active_reporters.pop(message.author.id)

    async def handle_channel_message(self, message):
        # handle messages sent in the "group-#" channel - eventually link classifier here
        if message.channel.name == f'group-{self.group_num}':
            # Forward the message to the mod channel
            mod_channel = self.mod_channels[message.guild.id]
            # await mod_channel.send(f'Forwarded message:\n{message.author.name}: "{message.content}"')
            scores = await self.eval_text(message)
            # await mod_channel.send(self.code_format(scores))
            return

        # handle moderator flow in mod channel
        elif message.channel.name == f'group-{self.group_num}-mod':
            if message.content == Report.QUEUE_KEYWORD:
                reply = "Moderation process started.\n\n"
                reply += "Here are the report queues and the number of messages in them:\n"

                for i in range(len(self.queue_num)):
                    reply += f"{i} - {self.queue_num[i]}: {len(self.reports[self.queue_num[i]])}\n"

                reply += "\nPlease enter the number for the queue you wish to work on."
                await message.channel.send(reply)
                return

            elif message.content == Report.MASS_KEYWORD:
                self.print_targetted_users_table()
                return

            # moderator choosing a message to address
            elif message.content.isnumeric():
                # if the moderator started another moderation before finishing the previous one, put that report back into queue.
                if message.author.id in self.active_moderators:
                    if self.active_moderators[message.author.id].report_csam():
                        self.reports['user_csam'].append(self.active_moderators[message.author.id])
                        self.active_moderators.pop(message.author.id)
                    elif self.active_moderators[message.author.id].report_adult():
                        self.reports['user_adult'].append(self.active_moderators[message.author.id])
                        self.active_moderators.pop(message.author.id)
                    else:
                        self.active_moderators.pop(message.author.id)

                idx = int(message.content)
                if not 0 <= idx < len(self.reports):
                    reply = "Please enter a valid queue number."
                    await message.channel.send(reply)
                    return
                elif len(self.reports[self.queue_num[idx]]) == 0:
                    reply = "This queue is currently empty. Please select another queue."
                    await message.channel.send(reply)
                    return
                else:
                    queue = self.reports[self.queue_num[idx]]
                    self.active_moderators[message.author.id] = queue.popleft()

                    # check if the current report has already been handled in the report table.
                    if self.in_report_table(self.active_moderators[message.author.id]):
                        # update tables
                        self.update_tables(self.active_moderators[message.author.id])
                        self.active_moderators.pop(message.author.id)

                        # check if we can assign the next report to the moderator
                        if len(queue) > 0:
                            self.active_moderators[message.author.id] = queue.popleft()
                        # if not, reply that the queue is empty
                        else:
                            reply = "The reports in this report queue has been handled. Please select another queue."
                            await message.channel.send(reply)
                            return

                    # designate current message being moderated
                    await self.mod_channel.send(f"Report checked out: \n{self.active_moderators[message.author.id].message.author}: `{self.active_moderators[message.author.id].message.content}`")
                    responses = await self.active_moderators[message.author.id].moderate(self.active_moderators[message.author.id].message)
                    for r in responses:
                        await message.channel.send(r)

            # if moderators message a non-queue word without selecting a queue
            elif message.author.id not in self.active_moderators:
                reply = "Please choose a report queue to start moderating."
                await message.channel.send(reply)

            elif message.content == Report.EXIT_KEYWORD:
                # if the moderator started another moderation before finishing the previous one, put that report back into queue.
                reply = "Moderation cancelled. The report has been reinserted into the reporting queue."

                if self.active_moderators[message.author.id].report_csam():
                    self.reports['user_csam'].append(self.active_moderators[message.author.id])
                    self.active_moderators.pop(message.author.id)
                elif self.active_moderators[message.author.id].report_adult():
                    self.reports['user_adult'].append(self.active_moderators[message.author.id])
                    self.active_moderators.pop(message.author.id)
                else:
                    self.active_moderators.pop(message.author.id)

                await message.channel.send(reply)
                return

            # moderator addressing a message
            elif message.content == "valid":
                if self.active_moderators[message.author.id].state == State.CSAM:
                    await self.mod_channel.send(f"Deleted by moderator: \n{self.active_moderators[message.author.id].message.author}: `{self.active_moderators[message.author.id].message.content}`")
                    await self.active_moderators[message.author.id].message.delete()
                    reply = "The message has been removed, the user has been banned, and NCMEC has been notified. Thank you!"
                    await message.channel.send(reply)

                    # update report table
                    self.add_to_report_table(self.active_moderators[message.author.id], "valid")
                    # update reporter table
                    self.update_reporter(self.active_moderators[message.author.id].reporter, 1, 0)
                    # clean up after moderation completed / prepare for moderating new report
                    self.active_moderators.pop(message.author.id)
                    return
                
                if self.active_moderators[message.author.id].state == State.ADULT:
                    await self.mod_channel.send(f"Deleted by moderator: \n{self.active_moderators[message.author.id].message.author}: `{self.active_moderators[message.author.id].message.content}`")
                    await self.active_moderators[message.author.id].message.delete()
                    offender = self.active_moderators[message.author.id].message.author
                    if offender in self.warned_users:
                        reply = "The message has been removed and the user has been banned. Thank you!"
                        await message.channel.send(reply)
                    else:
                        self.warned_users.add(offender)
                        reply = "The message has been removed and the user has been warned. Thank you!"
                        await message.channel.send(reply)

                    # update report table
                    self.add_to_report_table(self.active_moderators[message.author.id], "valid")
                    # update reporter table
                    self.update_reporter(self.active_moderators[message.author.id].reporter, 1, 0)
                    # clean up after moderation completed / prepare for moderating new report
                    self.active_moderators.pop(message.author.id)
                    return

            elif message.content == "wrong-type":
                if self.active_moderators[message.author.id].state == State.CSAM:
                    await self.mod_channel.send(
                        f"Deleted by moderator: \n{self.active_moderators[message.author.id].message.author}: `{self.active_moderators[message.author.id].message.content}`")
                    await self.active_moderators[message.author.id].message.delete()
                    offender = self.active_moderators[message.author.id].message.author
                    if offender in self.warned_users:
                        reply = "The message has been removed and the user has been banned. Thank you!"
                        await message.channel.send(reply)
                    else:
                        self.warned_users.add(offender)
                        reply = "The message has been removed and the user has been warned. Thank you!"
                        await message.channel.send(reply)

                    # update report table
                    self.add_to_report_table(self.active_moderators[message.author.id], "wrong-type")
                    # update reporter table
                    self.update_reporter(self.active_moderators[message.author.id].reporter, 1, 0)
                    # clean up after moderation completed / prepare for moderating new report
                    self.active_moderators.pop(message.author.id)
                    return

                if self.active_moderators[message.author.id].state == State.ADULT:
                    await self.mod_channel.send(
                        f"Deleted by moderator: \n{self.active_moderators[message.author.id].message.author}: `{message.content}`")
                    await self.active_moderators[message.author.id].message.delete()
                    reply = "The message has been removed, the user has been banned, and NCMEC has been notified. Thank you!"
                    await message.channel.send(reply)

                    # update report table
                    self.add_to_report_table(self.active_moderators[message.author.id], "wrong-type")
                    # update reporter table
                    self.update_reporter(self.active_moderators[message.author.id].reporter, 1, 0)
                    # clean up after moderation completed / prepare for moderating new report
                    self.active_moderators.pop(message.author.id)
                    return

            elif message.content == "invalid":
                # update report table
                self.add_to_report_table(self.active_moderators[message.author.id], "invalid")
                # update reporter table
                self.update_reporter(self.active_moderators[message.author.id].reporter, 0, 1)
                # update reportee table
                self.update_reportee(self.active_moderators[message.author.id].reportedUser,
                                     self.active_moderators[message.author.id].link, 0)
                # clean up after moderation completed / prepare for moderating new report
                self.active_moderators.pop(message.author.id)

                reply = "Report discarded. Thank you!"
                await message.channel.send(reply)
                return

    async def eval_text(self, message):
        ''''
        TODO: Once you know how you want to evaluate messages in your channel, 
        insert your code here! This will primarily be used in Milestone 3. 
        '''
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # don't evaluate message if there's no images
        if len(message.attachments) > 0:
            predictions = []
            for attachment in message.attachments:
                if attachment.content_type.startswith("image"):
                    img_bytes = await attachment.read()
                    img = Image.open(io.BytesIO(img_bytes))

                    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((32, 32), antialias=True),
                        transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),]
                    )

                    res = self.model(test_transform(img).unsqueeze(0))
                    predictions.append(classes[torch.argmax(res, dim=1)])
            
            if not message.content:
                message.content = ""
            # CSAM
            if "cat" in predictions:
                await self.add_message_to_bot_queue(message, State.CSAM)
            # Adult
            elif "dog" in predictions:
                await self.add_message_to_bot_queue(message, State.ADULT)
            
            await self.mod_channel.send(self.code_format(message.content))
        else:
            url = re.search("(?P<url>https?://[^\s]+)", message.content).group("url")
            if url:
                response = requests.get(url)
                img = Image.open(io.BytesIO(response.content))
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((32, 32), antialias=True),
                    transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)),]
                )
                res = self.model(test_transform(img).unsqueeze(0))
                prediction = classes[torch.argmax(res, dim=1)]
                print(prediction)

                # CSAM
                if prediction == "cat":
                    await self.add_message_to_bot_queue(message, State.CSAM)
                # Adult
                elif prediction == "dog":
                    await self.add_message_to_bot_queue(message, State.ADULT)
            await self.mod_channel.send(self.code_format(message.content))
        return
    
    def code_format(self, text):
        ''''
        TODO: Once you know how you want to show that a message has been 
        evaluated, insert your code here for formatting the string to be 
        shown in the mod channel. 
        '''
        return "Evaluated image: `" + text+ "`"

client = ModBot()
client.run(discord_token)
