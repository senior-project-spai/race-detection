import gender_classifier
import s3_helper
from PIL import Image
import os
from json import dumps
from kafka import KafkaProducer
from kafka import KafkaConsumer
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


KAFKA_HOST = os.environ['KAFKA_HOST']
KAFKA_PORT = os.environ['KAFKA_PORT']
KAFKA_TOPIC_INPUT = os.environ['KAFKA_TOPIC_INPUT']
KAFKA_TOPIC_OUTPUT = os.environ['KAFKA_TOPIC_OUTPUT']

# display environment variable
logger.info('KAFKA_HOST: {}'.format(KAFKA_HOST))
logger.info('KAFKA_PORT: {}'.format(KAFKA_PORT))
logger.info('KAFKA_TOPIC_INPUT: {}'.format(KAFKA_TOPIC_INPUT))
logger.info('KAFKA_TOPIC_OUTPUT: {}'.format(KAFKA_TOPIC_OUTPUT))


def main():
    consumer = KafkaConsumer(KAFKA_TOPIC_INPUT,
                             bootstrap_servers=[
                                 '{}:{}'.format(KAFKA_HOST, KAFKA_PORT)],
                             auto_offset_reset='earliest',
                             enable_auto_commit=True,
                             group_id='my-group')

    producer = KafkaProducer(
        bootstrap_servers=['{}:{}'.format(KAFKA_HOST, KAFKA_PORT)])

    logger.info('Ready for consuming messages')
    for message in consumer:
        # de-serialize
        inp = message.value.decode('utf-8')
        logger.info('Input path: {}'.format(inp))

        img_stream = s3_helper.get_file_stream_s3(inp)

        # Open image from stream
        img = Image.open(img_stream)

        # Show Image
        # plt.imshow(img)
        # plt.show()

        predict = gender_classifier.predict_one_image(img_stream)
        result = {'data': predict, 'filepath': inp}
        logger.info('Result: {}'.format(result))
        producer.send(KAFKA_TOPIC_OUTPUT, value=dumps(result).encode('utf-8'))


if __name__ == '__main__':
    main()
