import json
import os

# ── 3 hand-crafted AAC personas ───────────────────────────────────────────────
# Each has a distinct condition, voice, and bucketed memories.
# Depth > quantity: 3 rich personas beat 50 generic ones for retrieval quality.

PERSONAS = [

    {
        "profile": {
            "name":               "Mia Chen",
            "age":                28,
            "condition":          "cerebral palsy",
            "communication_style":"witty, dry humour, short punchy sentences, uses sarcasm",
            "access_method":      "webcam head-tracking",
            "languages":          ["English"]
        },
        "memory_buckets": {
            "family": [
                "My mom calls every Sunday and always asks if I've eaten. I love it but won't admit it.",
                "My brother Ravi helped me set up this AAC system. He's at Cornell doing CS.",
                "We do a family movie night every Diwali — always an 80s Bollywood film nobody likes except Dad.",
                "My parents moved from Chengdu before I was born. We still make dumplings on Chinese New Year.",
                "My sister Lena is three years younger and somehow already more responsible than me."
            ],
            "medical": [
                "I have a PT session every Tuesday at 2pm with Dr. Sandra Hollis.",
                "I use a power wheelchair. The joystick is on my left side.",
                "I'm allergic to penicillin. I have to mention this at every hospital visit.",
                "My spasticity is worse in cold weather. Winter in Chicago is not my friend.",
                "I use baclofen for muscle tone. It makes me sleepy if I take it too early."
            ],
            "hobbies": [
                "I follow competitive Smash Bros. I could beat most people if my hands worked differently.",
                "I've been watching every Studio Ghibli film in order. Currently on Porco Rosso.",
                "I collect vintage sci-fi paperbacks. Asimov and Le Guin mostly.",
                "I got really into chess puzzles during lockdown. Still do them before bed.",
                "I enjoy critiquing bad movie sequels. It's practically a hobby at this point."
            ],
            "daily_routine": [
                "Mornings are slow. I need about 45 minutes before I feel like a person.",
                "I order from the same Thai place every Friday. Green curry, always.",
                "I keep a voice memo journal since typing long things is tiring.",
                "I usually watch one episode of something after dinner to decompress.",
                "My caregiver Marcus arrives at 8am on weekdays. He makes decent coffee."
            ],
            "social": [
                "My best friend Priya visits on weekends. She narrates everything like a nature documentary.",
                "I'm part of an online disability advocacy group. We meet on Zoom every other Wednesday.",
                "I don't love big parties. Small dinners with three or four people are my ideal.",
                "My neighbour Tom always stops to chat when I'm outside. He's retired and lonely, I think.",
                "I met most of my close friends through a gaming Discord server."
            ]
        }
    },

    {
        "profile": {
            "name":               "Gerald Okafor",
            "age":                61,
            "condition":          "ALS (early-to-mid stage)",
            "communication_style":"formal, measured, eloquent, longer structured sentences",
            "access_method":      "eye-gaze device",
            "languages":          ["English"]
        },
        "memory_buckets": {
            "family": [
                "My wife Constance and I have been married for 34 years. She is the reason I stay organised.",
                "My son Emeka is a civil engineer based in Houston. He calls every Thursday evening.",
                "My daughter Adaeze is doing her residency in paediatrics in Baltimore. I am very proud.",
                "We used to take a family trip to Lagos every two years to visit my mother's side.",
                "My youngest grandchild, Tobenna, was born last April. I have not met him in person yet."
            ],
            "medical": [
                "I was diagnosed with ALS in November 2024. I am still adjusting to what that means day to day.",
                "My speech was the first thing to decline noticeably. That is why I began using AAC.",
                "I see my neurologist Dr. Patricia Eze at Northwestern every six weeks.",
                "I take riluzole daily. I have not noticed significant side effects so far.",
                "My occupational therapist is helping me adapt my home office for continued work."
            ],
            "hobbies": [
                "I taught economics at DePaul University for twenty-two years.",
                "I have read most of Chinua Achebe's work. Things Fall Apart shaped how I see storytelling.",
                "I enjoy chess — classical time controls, not blitz. Patience is the point.",
                "I used to cook elaborate Sunday stews. Constance has taken that over now, which is bittersweet.",
                "I listen to Fela Kuti when I need to feel grounded. Always has."
            ],
            "daily_routine": [
                "I begin each morning by reading two newspapers — the Tribune and the Guardian.",
                "I try to write for at least thirty minutes each day, even if it is just reflections.",
                "Afternoons are for rest. My energy is most reliable in the mornings.",
                "Constance and I watch the evening news together. We have done this for decades.",
                "I use the eye-gaze device for most communication now. It takes patience but it works."
            ],
            "social": [
                "My closest friend is Charles Nwosu. We have known each other since secondary school in Enugu.",
                "I stay in touch with former colleagues at DePaul, though visits have become less frequent.",
                "My church community at St. Clement has been a source of genuine support since my diagnosis.",
                "I prefer one-on-one conversations. I find group settings harder to follow now.",
                "I joined an ALS support group that meets virtually. It helps more than I expected."
            ]
        }
    },

    {
        "profile": {
            "name":               "Arjun Mehta",
            "age":                17,
            "condition":          "autism spectrum disorder (non-verbal)",
            "communication_style":"direct, topic-specific, narrow vocabulary, code-switches Hindi/English, routine-focused",
            "access_method":      "tablet touch grid + AAC app",
            "languages":          ["English", "Hindi"]
        },
        "memory_buckets": {
            "family": [
                "Mummy makes aloo paratha on Sunday mornings. That is my favourite thing.",
                "Papa works at a software company. He brings home a samosa sometimes on Fridays.",
                "My dadi lives with us. She watches serials very loudly but I like that she is home.",
                "My cousin Rohan visits in the summer. We play Minecraft together for many hours.",
                "Mummy knows what I want even when I cannot say it. She is very good at that."
            ],
            "medical": [
                "I see my therapist Riya didi every Wednesday at 4pm.",
                "I do not like the occupational therapy exercises but I do them.",
                "I cannot eat food that has a slimy texture. It makes me feel very bad.",
                "I take melatonin at night. Without it, sleeping is very hard.",
                "My school has a support aide named Mr. Fernandez. He is calm and that helps."
            ],
            "hobbies": [
                "I know the complete timetable of all Mumbai Metro lines.",
                "I like sorting my LEGO bricks by colour and size before building.",
                "My favourite YouTube channel is about deep sea creatures. Anglerfish are very strange.",
                "I have watched the same three episodes of Doraemon more than fifty times each.",
                "I am learning the capitals of every country. I know 142 so far."
            ],
            "daily_routine": [
                "I wake up at 6:47am. Changing this time makes my whole day feel wrong.",
                "I eat the same breakfast — two rotis with ghee and one glass of milk.",
                "School starts at 8:30am. I like to arrive before the other students.",
                "After school I need quiet time for at least one hour. No talking.",
                "Dinner must be at 7:30pm. If it is late I feel very unsettled."
            ],
            "social": [
                "I have one friend at school named Vivaan. We do not talk much but we sit together.",
                "I do not like it when people stand too close. One arm's distance is comfortable.",
                "I prefer typing to speaking when I need to say something important.",
                "Loud places with many people feel like too much information at once.",
                "I like it when people tell me exactly what is going to happen next."
            ]
        }
    }
]


def main():
    os.makedirs("memories", exist_ok=True)

    user_index = []

    for persona in PERSONAS:
        uid  = persona["profile"]["name"].lower().replace(" ", "_")
        path = f"memories/{uid}.json"

        with open(path, "w") as f:
            json.dump(persona, f, indent=2, ensure_ascii=False)

        user_index.append({
            "id":        uid,
            "name":      persona["profile"]["name"],
            "condition": persona["profile"]["condition"],
            "style":     persona["profile"]["communication_style"],
            "file":      path
        })

        print(f"  Wrote {path}")

    with open("users.json", "w") as f:
        json.dump({"users": user_index}, f, indent=2, ensure_ascii=False)

    print(f"\n Done — {len(PERSONAS)} personas written to memories/")
    print("  Files:", [u["file"] for u in user_index])


if __name__ == "__main__":
    main()