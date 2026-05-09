"""
live_feed.py
------------
Simulates a live tweet stream for demo purposes.
Generates new batches of tweets every call with realistic variation.
In production: swap get_live_tweets() with Twitter API v2 call.
"""

import random
import time
import pandas as pd
from datetime import datetime, timedelta

# ── Realistic disaster tweet pool ─────────────────────────────────────────────
TWEET_POOL = {
    "rescue": [
        "People trapped on rooftop in Srinagar, water level rising fast SOS",
        "3 families stranded in Silchar flood, rescue team not reached yet urgent help",
        "Helicopter needed in Kedarnath, 12 pilgrims cut off since 2 days",
        "Young child missing in flood water near Patna ghat, please help find",
        "Entire village submerged in Assam, residents need immediate evacuation",
        "Old couple stranded on second floor since morning, Bhubaneswar flood",
        "Rescue boat needed near Brahmaputra bank, 50 people waiting since dawn",
        "People stranded on highway due to Himachal landslide, no food or water",
        "SOS signal from fishermen near Gujarat coast, cyclone approaching fast",
        "150 tourists trapped in Leh after flash flood washed away road",
    ],
    "medical": [
        "Hospital in Silchar running out of blood supply, O+ needed urgently",
        "Cholera outbreak reported in Bihar flood relief camp, doctors needed",
        "Snake bite case in flood area Odisha, no antivenom available at local clinic",
        "Pregnant woman needs evacuation from Assam village, road blocked",
        "Dialysis patients in crisis after power outage at Patna hospital",
        "Field medical camp running low on medicines in Uttarakhand disaster zone",
        "Mass diarrhoea cases reported in Muzaffarpur flood shelter, urgent care",
        "5 injured in Manipur earthquake, nearest hospital 40 km away blocked",
        "Corona patient needs oxygen cylinder in flood hit area Surat",
        "Dengue cases rising fast in waterlogged areas of Mumbai suburbs",
    ],
    "food_water": [
        "No food since 2 days at relief camp near Guwahati, 300 families waiting",
        "Drinking water contaminated after flood in Varanasi slum area",
        "Children going hungry in shelter, Assam relief distribution not reached",
        "Fresh water packet distribution needed in 5 villages near Brahmaputra",
        "Food packets running out at Kaziranga relief centre, resupply urgently",
        "Baby formula not available in flood zone, infants at risk Odisha coast",
        "Relief trucks stuck at flooded road, 800 people without food today",
        "Rice and dal supply exhausted in Sylhet district camp, volunteers needed",
    ],
    "infrastructure": [
        "NH-44 blocked due to Himachal landslide, hundreds of vehicles stranded",
        "Brahmaputra bridge washed away, 3 districts completely isolated",
        "Power outage continues in entire Sylhet district after cyclone Mocha",
        "Mobile tower down in Kedarnath valley, communication completely cut",
        "Railway line submerged under flood near Muzaffarpur station",
        "Main road to Silchar under 4 feet water, no vehicles can pass",
        "Electricity poles fallen across highway in Gujarat after cyclone",
        "Dam overflow alert issued for districts downstream of Hirakud reservoir",
        "Internet and phone lines down in Manipur after 5.7 magnitude earthquake",
        "Airport runway flooded in Agartala, flights suspended till further notice",
    ],
    "information": [
        "IMD issues red alert for next 48 hours in Assam and Meghalaya",
        "NDRF 8 teams deployed across Bihar flood affected districts",
        "CM announces compensation for flood victims in Assam, Rs 3800 per family",
        "Official helpline number for flood rescue: 1078, available 24x7",
        "Army column from Tezpur reaching Silchar by evening with supplies",
        "Flood level in Brahmaputra showing slight decline says water commission",
        "School and colleges declared closed for 3 days in 5 flood hit districts",
        "National disaster management authority activates emergency protocol",
        "Satellite images show 40% of Kaziranga national park flooded",
        "Prime minister reviews flood situation in northeast, announces relief fund",
    ],
}

LOCATIONS = [
    "Chennai", "Assam", "Patna", "Bihar", "Uttarakhand",
    "Mumbai", "Odisha", "Gujarat", "Srinagar", "Silchar",
    "Manipur", "Kedarnath", "Guwahati", "Bhubaneswar", "Kolkata",
    "Himachal", "Rajasthan", "Lucknow", "Varanasi", "Nagpur",
]


def get_live_tweets(n: int = 20, seed: int = None) -> pd.DataFrame:
    """
    Returns a fresh batch of n simulated disaster tweets.
    seed=None means random every time (live feel).
    seed=int makes it reproducible for testing.
    """
    if seed is not None:
        random.seed(seed)

    all_tweets = []
    now = datetime.now()

    for i in range(n):
        category = random.choices(
            list(TWEET_POOL.keys()),
            weights=[0.30, 0.20, 0.18, 0.17, 0.15],  # rescue most common
            k=1
        )[0]

        tweet = random.choice(TWEET_POOL[category])
        location = random.choice(LOCATIONS)

        # Inject location variation
        if random.random() > 0.5 and location.lower() not in tweet.lower():
            tweet = tweet + f" #{location}"

        timestamp = now - timedelta(minutes=random.randint(0, 60))

        all_tweets.append({
            "text": tweet,
            "category_hint": category,
            "location_hint": location,
            "timestamp": timestamp,
            "source": "simulated_live_feed",
        })

    # Shuffle so it looks like a real mixed stream
    random.shuffle(all_tweets)
    df = pd.DataFrame(all_tweets)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def stream_tweets(batch_size: int = 10, interval_seconds: int = 5, max_batches: int = 6):
    """
    Generator that yields tweet batches every interval_seconds.
    Use this in a loop to simulate real-time streaming.
    """
    for i in range(max_batches):
        batch = get_live_tweets(n=batch_size)
        yield batch
        if i < max_batches - 1:
            time.sleep(interval_seconds)


if __name__ == "__main__":
    print("Fetching live tweet batch...")
    df = get_live_tweets(n=10)
    print(df[["text", "category_hint", "location_hint", "timestamp"]].to_string())
