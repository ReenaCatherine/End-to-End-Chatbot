#end to end chatbot

import os
import nltk
import ssl
import streamlit as st
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

#intents of chatbot
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame.", "To improve your credit score, pay your bills on time, reduce your debt, and avoid opening too many new credit accounts in a short period of time."]
    },
    {
        "tag": "location",
        "patterns": ["Where are you located", "Where is your office", "What is your address"],
        "responses": ["I exist in the digital world, so I don't have a physical location.", "I don't have a physical address, I'm a virtual assistant.", "You can find me here, online, anytime."]
    },
    {
        "tag": "hours",
        "patterns": ["What are your hours", "When are you available", "What time do you open"],
        "responses": ["I am available 24/7.", "You can reach out to me anytime.", "I'm always here to assist you."]
    },
    {
        "tag": "services",
        "patterns": ["What services do you offer", "How can you help me", "What can you assist with"],
        "responses": ["I can assist with answering questions, providing information, and helping with basic tasks.", "My services include general information, customer support, and assistance with common inquiries.", "I can help you find information, troubleshoot issues, and more."]
    },
    {
        "tag": "payment",
        "patterns": ["How do I make a payment", "What payment methods do you accept", "Can I pay online"],
        "responses": ["You can make a payment through our website using various methods like credit cards or PayPal.", "We accept credit cards, debit cards, and PayPal for online payments.", "To make a payment, visit our payment page and follow the instructions."]
    },
    {
        "tag": "refund",
        "patterns": ["How do I get a refund", "What is your refund policy", "Can I get my money back"],
        "responses": ["Our refund policy is available on our website. Please review it for details.", "To request a refund, contact our customer service team with your order details.", "Refunds are processed within 5-7 business days once approved."]
    },
    {
        "tag": "order_status",
        "patterns": ["What is the status of my order", "Where is my order", "Can I track my order"],
        "responses": ["You can check the status of your order by logging into your account and visiting the order history page.", "To track your order, use the tracking number provided in your confirmation email.", "For order status, visit our tracking page and enter your order number."]
    },
    {
        "tag": "shipping",
        "patterns": ["What are your shipping options", "How much is shipping", "Do you offer free shipping"],
        "responses": ["We offer various shipping options including standard and express delivery.", "Shipping costs vary based on your location and the shipping method chosen.", "Free shipping is available on orders over a certain amount. Check our website for details."]
    },
    {
        "tag": "product_info",
        "patterns": ["Tell me about this product", "What are the features of this item", "Is this product available"],
        "responses": ["You can find detailed product information on the product page.", "This item has several features including [feature1], [feature2], and [feature3].", "Availability of this product can be checked on the product page."]
    },
    {
        "tag": "appointment",
        "patterns": ["How do I book an appointment", "Can I schedule a meeting", "I need to make an appointment"],
        "responses": ["To book an appointment, visit our scheduling page and choose a time that works for you.", "You can schedule a meeting by contacting our support team or using our online booking system.", "Appointments can be made online through our booking portal."]
    },
    {
        "tag": "technical_support",
        "patterns": ["I need technical support", "How do I fix this issue", "Can you help me with a technical problem"],
        "responses": ["For technical support, please describe the issue you're facing.", "Our technical support team is here to help. What seems to be the problem?", "To fix this issue, you may try [solution1]. If that doesn't work, contact our support team."]
    },
    {
        "tag": "contact_info",
        "patterns": ["How can I contact you", "What is your phone number", "Do you have an email address"],
        "responses": ["You can contact us via email at support@example.com.", "Our phone number is (123) 456-7890.", "For inquiries, please email support@example.com or call us."]
    },
    {
        "tag": "feedback",
        "patterns": ["I want to give feedback", "Where can I leave a review", "How do I provide feedback"],
        "responses": ["We appreciate your feedback! You can leave a review on our website.", "To provide feedback, visit our feedback page.", "Your feedback helps us improve. Please share your thoughts on our review page."]
    },
    {
        "tag": "news",
        "patterns": ["What's the latest news", "Any updates", "What's new"],
        "responses": ["For the latest news and updates, visit our news section on the website.", "Check out our blog for recent updates and news.", "Stay updated with our latest news by visiting our website."]
    },
    {
        "tag": "joke",
        "patterns": ["Tell me a joke", "Make me laugh", "Do you know any jokes"],
        "responses": ["Why don't scientists trust atoms? Because they make up everything!", "What do you call fake spaghetti? An impasta!", "Why did the scarecrow win an award? Because he was outstanding in his field!"]
    },
    {
        "tag": "quote",
        "patterns": ["Give me a quote", "Inspirational quote", "Can you share a quote"],
        "responses": ["'The best way to predict the future is to create it.' - Peter Drucker", "'Success is not the key to happiness. Happiness is the key to success.' - Albert Schweitzer", "'The only limit to our realization of tomorrow is our doubts of today.' - Franklin D"]
    }
]

#Logistic Regression model

#vectorizer & classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state = 0, max_iter = 10000)

#data preprocessing
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

#training the model
X = vectorizer.fit_transform(patterns)
y = tags
clf.fit(X,y)

#function to chat with chatbot
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    #confidence score
    # tag_index = list(clf.classes_).index(tag)
    probabilities = clf.predict_proba(input_text)[0]
    # confidence_score = probabilities[tag_index]
    confidence_score = np.max(probabilities)
    #response
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response, confidence_score
        