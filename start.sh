#!/bin/bash
id=`ps aux|grep helmet.py|grep -v grep|awk '{print $2}'`
if [ -n $id ]
then
echo $id
kill -9 $id
fi
nohup python helmet.py > log.txt 2>&1 &
