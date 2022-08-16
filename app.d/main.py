import os
import json
import keras
import numpy as np
import requests
from slack_sdk import WebClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from threading import Thread
from websocket import create_connection, WebSocketConnectionClosedException

import deephaven.dtypes as dht
from deephaven import learn
from deephaven import DynamicTableWriter
from deephaven.table_listener import listen
from deephaven.learn import gather


SLACK_ENDPOINT = 'https://slack.com/api/apps.connections.open'

APP_TOKEN = os.environ["APP_TOKEN"]
BOT_OAUTH_TOKEN = os.environ["BOT_OAUTH_TOKEN"]
CHANNEL = os.environ["SLACK_CHANNEL_ID"]


# the max number of words in messages
# It can be decreased, if the most of messages are less than 200 words. Or it can be increased if you need more features in the model
MAX_NUMBER = 200

# types of toxic content we want to predict.
# Borrowed from Kaggle competition: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview
TOXICITY_TYPES = ["Toxic", "Severe_Toxic", "Obscene", "Threat", "Insult", "Identity_Hate"]


# call the apps.connections.open endpoint with app-level token to get a WebSocket URL
headers = {'Authorization': 'Bearer ' + APP_TOKEN, 'Content-type': 'application/x-www-form-urlencoded'}
response = requests.post(SLACK_ENDPOINT, headers=headers)
url = response.json()["url"]


# connect to the websocket
ws = create_connection(url)
ws.send(
    json.dumps(
        {
            "type": "subscribe",
            "token": BOT_OAUTH_TOKEN,
            "event": {
                "type": "message",
                "subtype": None
            }
        }
    )
)


# use Deephaven's DynamicTableWriter to create a table for features (integer representation of words)
# and original messages
columns = ["Index_" + str(num) for num in range(MAX_NUMBER)]
column_definitions = {col: dht.int32 for col in columns}
column_definitions["message"] = dht.string
dtw = DynamicTableWriter(column_definitions)
table = dtw.table



# use saved tokenizer from our pre-trained model
with open("/data/model/tokenizer.json") as json_file:
    tokenizer_json = json.load(json_file)
    tokenizer_json_string = json.dumps(tokenizer_json)
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json_string)

# load our pre-trained model
model = load_model("/data/model/model.h5")
print(model.summary())

# receive real-time messages by the websocket and write data to the table
def thread_function():
    while True:
        try:
            data = json.loads(ws.recv())
            event = data["payload"]["event"]
            message = event["text"]
            if (data["retry_attempt"] == 0 and "bot_id" not in event):
                # convert message into integer sequence encoding the words in the message
                list_tokenized = tokenizer.texts_to_sequences([message])
                row_to_write = pad_sequences(list_tokenized, maxlen=MAX_NUMBER)[0].tolist()
                row_to_write.append(message)
                # add integers representing words and original text to DH table
                dtw.write_row(*row_to_write)

        except Exception as e:
            print(e)


thread = Thread(target=thread_function)
thread.start()


# A function that gets the model's predictions on input data
def predict_with_model(features):
    predictions = model.predict(features)
    return predictions


# A function to gather data from table columns into a NumPy array of integers
def table_to_array_int(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type=np.intc)



def get_function_by_type(index):
    func = lambda data, idx: data[idx][index]
    return func

outputs = []  # list for learn.Output() objects
for i in range(len(TOXICITY_TYPES)):
    type = TOXICITY_TYPES[i]
    get_predicted_class = get_function_by_type(i)
    outputs.append(learn.Output(type, get_predicted_class, "double"))

# Use the learn function to create a new table that contains predicted values
predicted = learn.learn(
    table=table,
    model_func=predict_with_model,
    inputs=[learn.Input(columns, table_to_array_int)],
    outputs=outputs,
    batch_size=100
)

# use the Slack Web Client to send messages to a channel
client = WebClient(token=BOT_OAUTH_TOKEN)

# create a listener to our table with predictions.
# once the predicted table is updated, we post a warnig to our slack channel if the probability of toxicity > 0.5
# at least for one of indicators
def predicted_listener(update, is_replay):
    added_dict = update.added()
    warning = ""
    warning_types = [(type, added_dict[type]) for type in TOXICITY_TYPES if added_dict[type] > 0.5]
    for item in warning_types:
        warning += f'Detected: {item[0]} with probability {item[1][0]:.1f}. '
    if warning != "":
        client.chat_postMessage(channel=CHANNEL, text=warning)

predicted_handler = listen(predicted, predicted_listener)