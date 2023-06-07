#!/usr/bin/env bash
rm -rf ./cs152bots
sudo apt-get update
sudo apt install python3-pip
git clone https://github.com/stilakid/cs152bots.git
pip install discord.py
pip install requests
pip install cityhash
pip install gdown

gdown https://drive.google.com/uc?id=1R4nxBkL9mvz5qgfPs4ngyPDF7yaRJvqw