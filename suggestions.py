# def suggesstions(character):
    
#     word_dict = {
#         "A": ["Available", "Appreciate", "Awesome", "All the best", "At your service"],
#         "B": ["Best regards", "Busy now", "Be right back", "By the way", "Back soon"],
#         "C": ["Call me", "Catch you later", "Cheers", "Can we talk?", "Cool"],
#         "D": ["Do not disturb", "Done", "Definitely", "Don't worry", "Don't forget"],
#         "E": ["Excellent", "Excuse me", "Everything is fine", "Easy", "Enjoy"],
#         "F": ["For your information", "Fine", "Fantastic", "Feel free", "Follow up"],
#         "G": ["Good morning", "Great", "Got it", "Good evening", "Good luck"],
#         "H": ["Hi", "Hello", "Hope you’re doing well", "How are you?", "Have a great day"],
#         "I": ["I’m here", "I agree", "In progress", "I’m available", "I’m sorry"],
#         "J": ["Just a moment", "Just checking in", "Join me", "Just kidding", "Jump in"],
#         "K": ["Keep going", "Know this", "Keep me posted", "Kind regards", "Keep it up"],
#         "L": ["Let me know", "Looking forward", "Let’s go", "Long time no see", "Love it"],
#         "M": ["Meet me", "Much appreciated", "My bad", "Miss you", "More details"],
#         "N": ["No problem", "Noted", "Next time", "No worries", "Nice to meet you"],
#         "O": ["Okay", "On my way", "Oh no", "Out now", "One moment"],
#         "P": ["Please", "Perfect", "Pretty good", "Please confirm", "Proceed"],
#         "Q": ["Quick question", "Quite busy", "Questions?", "Quiet now", "Quickly done"],
#         "R": ["Right away", "Relax", "Rest assured", "Reach out", "Really good"],
#         "S": ["See you", "Sounds good", "Sure", "Stay safe", "So sorry"],
#         "T": ["Thank you", "Talk later", "Take care", "The best", "That’s fine"],
#         "U": ["Understood", "Unbelievable", "Up to you", "Urgent", "Until later"],
#         "V": ["Very good", "View it", "Virtually there", "Very well", "Verify"],
#         "W": ["What’s up", "Will do", "Well done", "Welcome", "Work in progress"],
#         "X": ["XOXO", "eXcellent idea", "eXactly", "eXpect more", "eXtra help"],
#         "Y": ["Yes", "You’re welcome", "You too", "Your call", "Yes please"],
#         "Z": ["Zoom meeting", "Zero worries", "Zip it", "Zeal", "Zoned out"]
#     }

#     return word_dict.get(character.upper(), [])


from transformers import pipeline

# Load GPT-based predictor for next-word suggestions
predictor = pipeline("text-generation", model="gpt2", max_length=10)

def suggestions(typed_word):
    # Static predefined suggestions
    word_dict = {
        "A": ["Available", "Appreciate", "Awesome", "All the best", "At your service"],
        "B": ["Best regards", "Busy now", "Be right back", "By the way", "Back soon"],
        "C": ["Call me", "Catch you later", "Cheers", "Can we talk?", "Cool"],
        "D": ["Do not disturb", "Done", "Definitely", "Don't worry", "Don't forget"],
        "E": ["Excellent", "Excuse me", "Everything is fine", "Easy", "Enjoy"],
        "F": ["For your information", "Fine", "Fantastic", "Feel free", "Follow up"],
        "G": ["Good morning", "Great", "Got it", "Good evening", "Good luck"],
        "H": ["Hi", "Hello", "Hope you’re doing well", "How are you?", "Have a great day"],
        "I": ["I’m here", "I agree", "In progress", "I’m available", "I’m sorry"],
        "J": ["Just a moment", "Just checking in", "Join me", "Just kidding", "Jump in"],
        "K": ["Keep going", "Know this", "Keep me posted", "Kind regards", "Keep it up"],
        "L": ["Let me know", "Looking forward", "Let’s go", "Long time no see", "Love it"],
        "M": ["Meet me", "Much appreciated", "My bad", "Miss you", "More details"],
        "N": ["No problem", "Noted", "Next time", "No worries", "Nice to meet you"],
        "O": ["Okay", "On my way", "Oh no", "Out now", "One moment"],
        "P": ["Please", "Perfect", "Pretty good", "Please confirm", "Proceed"],
        "Q": ["Quick question", "Quite busy", "Questions?", "Quiet now", "Quickly done"],
        "R": ["Right away", "Relax", "Rest assured", "Reach out", "Really good"],
        "S": ["See you", "Sounds good", "Sure", "Stay safe", "So sorry"],
        "T": ["Thank you", "Talk later", "Take care", "The best", "That’s fine"],
        "U": ["Understood", "Unbelievable", "Up to you", "Urgent", "Until later"],
        "V": ["Very good", "View it", "Virtually there", "Very well", "Verify"],
        "W": ["What’s up", "Will do", "Well done", "Welcome", "Work in progress"],
        "X": ["XOXO", "eXcellent idea", "eXactly", "eXpect more", "eXtra help"],
        "Y": ["Yes", "You’re welcome", "You too", "Your call", "Yes please"],
        "Z": ["Zoom meeting", "Zero worries", "Zip it", "Zeal", "Zoned out"]
    }

    # dynamic suggestions
    static_suggestions = word_dict.get(typed_word[0].upper(), [])
    
    context = " ".join(typed_word.strip().split()[-3:])
    context_predictions = predictor(context)[0]['generated_text'].split()

    return list(set(static_suggestions + context_predictions))[:5]  

