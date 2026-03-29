def classify(query: str) -> dict:
    q = query.lower()
    
    # contextual signals
    if any(w in q for w in ["you just said", "what did you say", "earlier", "you mentioned"]):
        return {"intent": "contextual", "bucket": None}
    
    # open domain signals  
    if any(w in q for w in ["capital of", "who is", "what is", "define", "explain"]):
        return {"intent": "open-domain", "bucket": None}
    
    # bucket hints
    if any(w in q for w in ["medication", "medicine", "doctor", "health", "allergic", "therapy"]):
        return {"intent": "personal", "bucket": "medical"}
    if any(w in q for w in ["family", "mom", "dad", "brother", "sister", "parents"]):
        return {"intent": "personal", "bucket": "family"}
    if any(w in q for w in ["hobby", "like to do", "enjoy", "weekend", "fun"]):
        return {"intent": "personal", "bucket": "hobbies"}
    if any(w in q for w in ["routine", "morning", "wake", "sleep", "daily"]):
        return {"intent": "personal", "bucket": "daily_routine"}
    if any(w in q for w in ["friend", "social", "people", "party", "community"]):
        return {"intent": "personal", "bucket": "social"}
    
    # default
    return {"intent": "personal", "bucket": None}