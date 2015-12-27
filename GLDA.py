"""
Perform LDA (Latent Dirichlet Allocation) on a gmail inbox
to cluster e-mails into topics (topic modeling)

Starter code for validating credentials from:
    https://developers.google.com/gmail/api/quickstart/python

Author: Darshan Thaker
"""
#!/usr/bin/python

from __future__ import print_function
import httplib2
import os
import pprint
import base64
import time
import sys
import threading

from apiclient import discovery
from gensim import corpora, models, similarities
from collections import defaultdict
import oauth2client
from oauth2client import client
from oauth2client import tools

try:
    import argparse
    parser = argparse.ArgumentParser(parents=[tools.argparser])
    parser.add_argument("nWorkers", type=int)
    parser.add_argument("preExisting", type=int)
    newFlags = parser.parse_args()
    # These flags are for getCredentials function for Gmail API
    flags = parser.parse_args()
    del flags.nWorkers
    del flags.preExisting
except ImportError:
    flags = None

"""
    My barrier implementation for Python2.7
"""
class Barrier:
    """
        Constructor for Barrier
        Input: n for number of threads that need to be
        synchronized.
    """
    def __init__(self, n):
        self.togo = n
        self.sem = threading.Semaphore(0)
        self.mutex = threading.Semaphore(1)
        self.count = 0
    
    """
        Make sure all threads reach sync() before continuing.
        Block until all threads have reached this point.
    """
    def sync(self):
        self.mutex.acquire()
        self.count += 1
        self.mutex.release()
        if (self.count < self.togo):
            self.sem.acquire()
        self.sem.release()

SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Gmail Topic Modeling'
DICTIONARY_FILE = 'gmail.dict'
CORPUS_FILE = 'gmailCorpus.mm'
MODEL_FILE = 'LDAmodel'
NUM_WORKERS = int(newFlags.nWorkers)
PREEXISTING = bool(newFlags.preExisting)
QUERY = 'label:inbox'
# Barrier for NUM_WORKERS threads + 1 main thread
barrier = Barrier(NUM_WORKERS + 1)

def getCredentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-python-quickstart.json')

    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

"""
    Worker thread that appends to the final list parameter
    final[0] = texts
    final[1] = corpus
    Input is a start and end index into the messages list
"""
def dataWorker(start, end, final, messages):
    documents = []
    credentials = getCredentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    query = QUERY

    for i in range(start, end):
        msg_id = messages[i]['id']
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()

        if 'multipart' in str(message['payload']['mimeType']):
            parts = message['payload']['parts']
        else:
            parts = [message['payload']]
            #pp = pprint.PrettyPrinter(indent=4)
            #pp.pprint(message)

        for content in parts:
            if content['mimeType'] == 'text/plain':
                try:
                    fullMessage = base64.urlsafe_b64decode(str(content['body']['data']))
                    fullMessage = fullMessage.decode('unicode-escape', 'ignore')
                    documents.append(fullMessage)
                    #print(base64.urlsafe_b64decode(str(content['body']['data'])))
                except KeyError:
                    continue

    print("Removing stopwords...")
    stoplist = set(("for a an this under through are them with you got we that be as our" +
                    "have your of what is his her on at and or to in not aren't when \r \n").split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
                 for document in documents] 
    startlist = tuple('> http - ~ = the [ 1 2 3 4 5 6 7 8 9 0 from www ..'.split())
    texts = [[word for word in text if not word.startswith(startlist)] for text in texts]

    # Remove words that appear only once
    print("Removing words that appear only once...")
    frequency = defaultdict(int)
    for text in texts:
       for token in text:
           frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1]
            for text in texts] 
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    final.append(texts)
    final.append(corpus)
    barrier.sync()

"""
    Reads the list of messages that match the given query from the Gmail API
    Spawns NUM_WORKERS threads that operate on partitions of the messages list.
"""
def prepareData():
    credentials = getCredentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    query = QUERY

    response = service.users().messages().list(userId='me', q=query).execute()

    messages = []
    if 'messages' in response:
        messages.extend(response['messages']) 

    while 'nextPageToken' in response:
        page_token = response['nextPageToken']
        response = service.users().messages().list(userId='me', q=query,
                                         pageToken=page_token).execute()
        messages.extend(response['messages'])

    #print("len(messages) = %d" % (len(messages)))
    textsSub = [[] for x in range(0, NUM_WORKERS)]

    start = time.time()
    for i in range(0, NUM_WORKERS):
        #Partition the messages list to each thread
        low = (i * len(messages) / NUM_WORKERS)
        if (i == NUM_WORKERS - 1):
            high = len(messages)
        else:
            high = ((i + 1) * len(messages)/NUM_WORKERS)
        
        thread = threading.Thread(target=dataWorker, args=(low, high, textsSub[i], messages))
        thread.start()

    # Use a barrier for synchronization to make sure all threads have finished
    barrier.sync()
    # Merge all the texts and corpuses generated from each thread into one large texts and corpus list
    texts = [x for text in textsSub for x in text[0]]
    corpus = [x for corp in textsSub for x in corp[1]]
    #pprint.pprint(textsSub)

    print("Serializing dictionary...")
    dictionary = corpora.Dictionary(texts)
    dictionary.save(DICTIONARY_FILE)

    print("Serializing corpus...")
    corpora.MmCorpus.serialize(CORPUS_FILE, corpus)
    print("Total took %d seconds" % (time.time() - start))
            
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(message)

def main():
    cwd = os.path.expanduser('.')
    dict_dir = os.path.join(cwd, DICTIONARY_FILE)
    corpus_dir = os.path.join(cwd, CORPUS_FILE)
    if not os.path.exists(dict_dir) or not os.path.exists(corpus_dir) or not PREEXISTING:
        if PREEXISTING:
            print("Selected 'use preExisting model', but doesn't exist")
        else:
            print("Selected 'do not use preExisting model'")
        # Only generate texts and corpus if needed or prompted by user.
        prepareData()
    
    print("Loading dictionary...")
    id2word = corpora.Dictionary.load(DICTIONARY_FILE)
    print("Loading corpus...")
    corpus = corpora.MmCorpus(CORPUS_FILE)

    print("Running LDA...")
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,
                                   num_topics=100, update_every=1,
                                   chunksize=10000, passes=1)
    print(lda.print_topics(20))
    print("Saving model...")
    lda.save(MODEL_FILE)
    
if __name__ == '__main__':
    main()
