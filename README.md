# toxic-message-classification

This app receives incoming messages from a Slack channel, stores this real-time chat data in a Deephaven streaming table, predicts a probability of toxicity for each message, and sends a notification if a message is classified as toxic.

## Components

### General

* [`start.sh`](start.sh) - A helper script to launch the application.
* [`docker-compose.yml`](docker-compose.yml) - The Docker compose file that defines the Deephaven images.
* [`Dockerfile`](Dockerfile) - The Dockerfile for the application. Simply extends Deephaven's base image with dependencies and app-mode scripts.
* [`requirements.txt`](requirements.txt) - The Python dependencies for the app.

### Files

* [`main.py`](app.d/main.py) - main Python script
* [`app.app`](app.d/app.app) - The Deephaven App Mode config file


## High level overview

This app checks if a new message posted to a Slack channel reads as toxic. If so, a bot sends a warning message to the channel.

The process contains 3 steps:

 - receive slack events using the Slack Real Time Messaging API and store real-time chat data in a Deephaven streaming table
 - predict a probability of toxicity for each message
 - send a notification if a message is classified as toxic

## How to run
Run 'sh start.sh' and open http://localhost:10000/ide/ 