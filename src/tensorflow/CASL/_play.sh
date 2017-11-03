#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Doorpuzzle python module                                                                              
print_header "Installing Doorpuzzle python module"
cd $DIR/../../environment/Doorpuzzle
sudo pip install -I .

# Mazeworld python module                                                                              
print_header "Installing Minecraft python module"
cd $DIR/../../environment/Minecraft
sudo pip install -I .

# Play
print_header "Playing using Pre-trained Network"
cd $DIR
python CASL.py PLAY_MODE=True
