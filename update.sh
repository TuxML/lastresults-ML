#!/bin/bash

DIRECTORY=$1
echo $DIRECTORY
mkdir $DIRECTORY
IGRIDA_PATH="macher@igrida-oar-frontend:/temp_dd/igrida-fs1/macher/analysisrf/rf-analysis2/rf-analysis/$DIRECTORY/*"
scp "macher@igrida-oar-frontend:/temp_dd/igrida-fs1/macher/analysisrf/rf-analysis2/rf-analysis/$DIRECTORY/*" $DIRECTORY
scp macher@igrida-oar-frontend:/temp_dd/igrida-fs1/macher/analysisrf/rf-analysis2/rf-analysis/config/config.json $DIRECTORY