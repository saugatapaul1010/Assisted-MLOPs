#!/bin/bash
########################################################
## Shell Script to Build Docker Image by Anuj Shah 
## script not completed - Do not use  
########################################################
CONTAINER_NAME="integrator-amlops-test"
IMAGE_NAME="integrator-amlops-test"
DATE=`date +%Y.%m.%d.%H.%M`
echo "skip building? (y/n)"
old_stty_cfg=$(stty -g)
stty raw -echo
answer=$( while ! head -c 1 | grep -i '[ny]' ;do true ;done )
stty $old_stty_cfg
if echo "$answer" | grep -iq "^y";then
    echo "Skipped building"
else
    result=$( sudo docker ps -qf "name=$CONTAINER_NAME*" )
    if [[ -n "$result" ]]; then
        sudo docker ps -qf "name=$CONTAINER_NAME*" | awk '{ print $1 }' | sudo docker rm -f $(</dev/stdin)
        # result=$( sudo docker ps -a | grep "integrator-amlops*" )
        # if [[ -n "$result" ]]; then
        #     echo "Removing Container"
        #     sudo docker ps -a | grep "integrator-amlops*" | awk '{ print $1 }' | sudo docker container rm $(</dev/stdin)
        #     echo "Deleted the existing docker container"
        # fi
    else
        echo "No Container with name $CONTAINER_NAME"
    fi
    result=$( sudo docker images -q $IMAGE_NAME )
    if [[ -n "$result" ]]; then
        echo "image exists"
        sudo docker images -q $IMAGE_NAME  | awk '{print $1}' | awk 'NR==1' | sudo docker rmi $(</dev/stdin)
    else
        echo "No such image"
    fi
    echo "build the docker image"
    sudo docker build -t $IMAGE_NAME:$DATE . --network="host">> build.out
    echo "build complete"
    echo "Running the updated image"
    sudo docker run --network="host" --restart=always --name=$CONTAINER_NAME -it -p 8000:8000 $IMAGE_NAME:$DATE
fi