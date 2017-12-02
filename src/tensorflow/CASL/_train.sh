#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}


# Check for libtcmalloc_minimal.so.4 
if [[ ! -f /usr/lib/libtcmalloc_minimal.so.4 ]] 
then
    echo 'File "/usr/lib/libtcmalloc_minimal.so.4 does not exist, aborting. Make sure you ran dependencies_install.sh if using ALE code!'
    exit
else
    if [[ $LD_PRELOAD != *"/usr/lib/libtcmalloc_minimal.so.4"* ]]; then
        echo '$LD_PRELOAD does not contain "/usr/lib/libtcmalloc_minimal.so.4", aborting. Make sure you ran dependencies_install.sh if using ALE code!'
        exit
    fi
fi

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Gridworld-Audio python module                                                                              
print_header "Installing Gridworld-Audio python module"
cd $DIR/../../environment/Gridworld-Audio
sudo pip install -I .

# SDworld python module                                                                              
print_header "Installing SDworld python module"
cd $DIR/../../environment/SDworld
sudo pip install -I .

# SDworld-hard python module                                                                              
print_header "Installing SDworld_hard python module"
cd $DIR/../../environment/SDworld_hard
sudo pip install -I .

# Train tf 
print_header "Training network"
cd $DIR
python GA3C.py "$@"
