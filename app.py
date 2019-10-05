from kafka import KafkaConsumer
from kafka import KafkaProducer
from json import dumps
import os

# Image
# import matplotlib.pyplot as plt
from PIL import Image

# local
import s3_helper
import gender_classifier

KAFKA_HOST = os.environ['KAFKA_HOST']
KAFKA_PORT = os.environ['KAFKA_PORT']
KAFKA_TOPIC_INPUT = os.environ['KAFKA_TOPIC_INPUT']
KAFKA_TOPIC_OUTPUT = os.environ['KAFKA_TOPIC_OUTPUT']


def main():
    consumer = KafkaConsumer(KAFKA_TOPIC_INPUT,
                             bootstrap_servers=['{}:{}'.format(KAFKA_HOST, KAFKA_PORT)],
                             auto_offset_reset='earliest',
                             enable_auto_commit=True,
                             group_id='my-group')
    
    producer = KafkaProducer(bootstrap_servers=['{}:{}'.format(KAFKA_HOST, KAFKA_PORT)])

    for message in consumer:
        # de-serialize
        inp = message.value.decode('utf-8')

        img_stream = s3_helper.get_file_stream_s3(inp)

        # Open image from stream
        img = Image.open(img_stream)
    
        # Show Image
        # plt.imshow(img)
        # plt.show()

        result = gender_classifier.predict_one_image(img_stream)
        producer.send(KAFKA_TOPIC_OUTPUT, value=dumps(result).encode('utf-8'))



if __name__ == '__main__':
    main()
