
# ISL

ISL is a project to enhance the communication capabilities of people with hearing abilities.

## Introduction
Sign language is a visual means of communicating through hand signals, gestures, facial expressions, and body language.
It’s the main form of communication for the Deaf and Hard-of-Hearing community, but sign language can be useful for other groups of people as well. 

ISL(Indian Sign Language) is predominantly used in Indian subcontinent. It is used by at least 7 million deaf signers.

The model takes a sentence(voice/text) as input and displays an ISL representative video of the sentence, keeping in mind all the rules and grammar of ISL. 

This model have been implemented by using two different parsers. ISL benepar.py uses benepar parser while model.py uses CoreNLPparser.

The goal is to help the deaf community by giving them resources by using which they don't feel inferior and get facilties so they can unleash their full potential and their disability be no more the obstacle between them and their dreams.

## Dataset

The videos of the dataset were downloaded from https://indiansignlanguage.org/ and then processed to make the final dataset.
Currently, our dataset consists of more than 300 words.

## Features

• It's first of it's kind which uses real persons instead of animations in displayed video.

• Can process voice both with internet and without internet.

• Uses synonyms of words, which are selected manually, so that if any word is used outside of dataset but has a word similar to it in our dataset, then it will be replaced by the word similar to it present in our dataset.

• If any word or a word similar to it is not present in our dataset, then it's letter by letter representation is displayed.

• The model keeps all rules of ISL grammar in mind while conversion of a sentence from english to it's ISL representation.

• Can handle short forms and contractions of words too.
