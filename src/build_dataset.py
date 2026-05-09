"""
build_dataset.py
----------------
Generates a realistic HumAID-style disaster tweet dataset locally.
NO internet required. NO API keys. Runs in seconds.

Produces 7000+ tweets across:
  - 12 real disaster event types
  - 10 HumAID label categories
  - 30 Indian locations
  - Varied, non-repetitive text using templates + random fill

Run once to create: outputs/humaid_local.csv
Then pipeline.py uses it automatically.
"""

import os
import random
import pandas as pd

CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "humaid_local.csv"
)
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

# ── Events (real disaster names for authenticity) ─────────────────────────
EVENTS = [
    "2015_nepal_earthquake",       "2017_hurricane_harvey",
    "2017_hurricane_irma",         "2016_louisiana_floods",
    "2013_typhoon_yolanda",        "2014_india_floods",
    "2017_mexico_earthquake",      "2016_hurricane_matthew",
    "2022_pakistan_floods",        "2023_turkey_earthquake",
    "2022_assam_floods",           "2023_odisha_cyclone",
]

LABELS = [
    "rescue_volunteering_or_donation_effort",
    "requests_or_urgent_needs",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "displaced_people_and_evacuations",
    "caution_and_advice",
    "other_relevant_information",
    "sympathy_and_support",
    "missing_or_found_people",
    "affected_individuals",
]

LABEL_TO_URGENCY = {
    "rescue_volunteering_or_donation_effort": "rescue",
    "requests_or_urgent_needs":               "rescue",
    "infrastructure_and_utility_damage":      "infrastructure",
    "injured_or_dead_people":                 "medical",
    "displaced_people_and_evacuations":       "rescue",
    "caution_and_advice":                     "information",
    "other_relevant_information":             "information",
    "sympathy_and_support":                   "information",
    "missing_or_found_people":                "rescue",
    "affected_individuals":                   "rescue",
}

# ── Text templates ────────────────────────────────────────────────────────
TEMPLATES = {
    "rescue_volunteering_or_donation_effort": [
        "Volunteers needed at {loc} relief camp, donations of food and clothes welcome",
        "Local NGO organizing rescue operations in {loc}, need 50 volunteers urgently",
        "Blood donation camp organized near {loc} hospital for flood victims",
        "Relief material being collected at {org} headquarters for {event} victims",
        "Rescue boats deployed in {loc} after flood, 200 families helped so far",
        "NDRF team from {loc} airlifted {num} people from rooftops today",
        "Community kitchen serving {num} meals daily at {loc} relief shelter",
        "Donation drive at {loc} collecting dry ration for displaced families",
        "Army engineers repairing {loc} bridge to restore supply chain for relief workers",
        "Medical volunteers from {org} reached {loc} flood affected villages with medicines",
        "Crowdfunding campaign launched for {loc} earthquake victims, target 10 lakh",
        "Relief trucks loaded with essentials leaving {org} depot for {loc} tonight",
        "Students from {loc} college forming human chain to help flood rescue teams",
    ],
    "requests_or_urgent_needs": [
        "Urgent: family of 5 stranded at {loc}, water level rising, need rescue boat",
        "SOS from {loc}: elderly person needs medical evacuation, no transport available",
        "Rescue needed at {loc} ghat, 3 children stuck on rooftop since morning",
        "Food and water urgently needed at {loc} shelter, {num} families waiting since dawn",
        "Medical supplies running out at {loc} field hospital, need immediate replenishment",
        "Generator fuel needed at {loc} relief camp, power has been down {num} days",
        "Baby formula urgently needed in {loc} flood zone for infants at shelter",
        "Dialysis patient in {loc} needs emergency transport to hospital, road flooded",
        "Oxygen cylinders needed at {loc} medical camp, running critically low",
        "Boat needed at {loc} to reach villages completely cut off by floodwater",
        "Help: {num} senior citizens trapped at {loc} old age home, need evacuation now",
        "Pregnant woman in {loc} needs urgent medical help, nearest hospital inaccessible",
        "No drinking water in {loc} for {num} days, tankers not arrived yet please help",
    ],
    "infrastructure_and_utility_damage": [
        "{loc} bridge completely washed away, 4 villages isolated from relief supplies",
        "Power lines down across {loc} district after cyclone, {num} days no electricity",
        "NH-{num} blocked near {loc} due to landslide, traffic diverted via longer route",
        "Mobile towers down in {loc} valley, communication completely cut off",
        "Railway track submerged at {loc} station, trains suspended till further notice",
        "Flood water entered {loc} airport runway, all operations suspended indefinitely",
        "Water supply pipes damaged in {loc}, {num} thousand residents without drinking water",
        "Road connecting {loc} to district headquarters under 4 feet of water, impassable",
        "Dam near {loc} overflowing, downstream villages on high alert for evacuation",
        "Internet cables damaged in {loc} area, no connectivity for {num} days",
        "Major overbridge near {loc} cracked after earthquake, declared unsafe for vehicles",
        "Gas pipeline rupture reported near {loc} after tremors, area being evacuated",
    ],
    "injured_or_dead_people": [
        "{num} people injured in {loc} after building collapse due to earthquake tremors",
        "Death toll in {loc} floods rises to {num}, rescue operations still ongoing",
        "{num} missing after {loc} landslide, search operations continuing in dark",
        "Casualty reported near {loc} after flash flood swept vehicle into river",
        "{num} bodies recovered from {loc} debris after cyclone, families informed",
        "Snake bite cases rising sharply in {loc} flood zones due to displaced reptiles",
        "Cardiac patient in {loc} relief camp collapsed, needs urgent hospitalisation",
        "{num} people hospitalised in {loc} after consuming contaminated floodwater",
        "Young child critically injured in {loc} wall collapse after heavy rains",
        "Fisherman from {loc} still missing after boat capsized in cyclone, day {num}",
    ],
    "displaced_people_and_evacuations": [
        "Over {num} families evacuated from {loc} as river crosses danger mark overnight",
        "Evacuation order issued for {loc} riverside villages, government buses arranged",
        "{num} people shifted to relief camps in {loc} ahead of cyclone expected landfall",
        "Temporary shelter set up at {loc} school for {num} displaced families",
        "Elderly and children evacuated first from {loc} flood affected low-lying areas",
        "Migrants stranded at {loc} station, special trains being arranged by railways",
        "{num} pilgrims evacuated from {loc} after road blocked by landslide",
        "Entire {loc} village relocated to higher ground as river reaches danger level",
        "Mass evacuation of coastal {loc} village begins, Navy assists in operation",
        "Relief camps at {loc} housing {num} people now, more being added urgently",
    ],
    "caution_and_advice": [
        "IMD issues red alert for {loc} district, very heavy rainfall expected next 48 hours",
        "Residents near {loc} river warned to shift to higher ground immediately tonight",
        "Do not use road near {loc} bridge, structure damaged and unsafe after flood",
        "Boil water before drinking in {loc} area due to severe flood contamination risk",
        "Stay indoors in {loc} as cyclone expected to make landfall early tomorrow morning",
        "Advisory: avoid travel to {loc} hills due to active landslide zone after rains",
        "NDMA warns {loc} residents: river levels rising rapidly, evacuate low lying areas",
        "Fishermen advised not to venture into sea near {loc} coast for next 72 hours",
        "Parents advised to keep children away from {loc} waterlogged streets and drains",
        "Do not ignore evacuation notices in {loc}, last warning before gates closed",
    ],
    "other_relevant_information": [
        "Chief Minister {loc} state reviews flood situation, announces Rs 500 crore relief fund",
        "NDRF teams deployed in {num} districts of {loc} state as situation worsens",
        "Satellite images reveal extent of damage in {loc} flood affected coastal region",
        "Relief helpline number for {loc} district: 1078, operational round the clock",
        "Weather improving in {loc}, flood waters slowly receding says water resources dept",
        "Compensation announced for flood victims in {loc}: Rs 4000 per affected family",
        "Schools closed for {num} days in {loc} district due to ongoing flood situation",
        "Army helicopter makes {num} sorties over {loc} dropping relief materials today",
        "Prime Minister speaks to {loc} Chief Minister, assures full central support",
        "Flood situation in {loc} improving, water level dropped {num} cm in last 24 hours",
    ],
    "sympathy_and_support": [
        "Prayers for all families affected by the devastating floods in {loc} right now",
        "Heart goes out to the people of {loc} dealing with this terrible tragedy",
        "Thinking of everyone impacted by the {loc} earthquake, stay strong and safe",
        "Solidarity with {loc} flood victims, hope rescue and relief reaches you soon",
        "Watching the news from {loc} with a heavy heart, hoping for quick recovery",
        "Standing with the brave people of {loc} through this disaster, you are not alone",
        "Sending strength and support to all relief workers in {loc}, thank you for service",
    ],
    "missing_or_found_people": [
        "Missing: person last seen near {loc} river, please contact district helpline",
        "Found: unidentified elderly man near {loc} relief camp, contact local authorities",
        "Searching for family from {loc} village, missing since flood day, any information",
        "{num} missing persons from {loc} area found safe at government relief camp",
        "Appeal: help locate family from {loc} coastal area, missing after cyclone",
        "Lost contact with relatives in {loc} flood zone since {num} days, very worried",
        "{num} missing fishermen from {loc} found alive on stranded boat, navy rescued",
    ],
    "affected_individuals": [
        "Lost everything in {loc} flood, government please help us rebuild our home",
        "{loc} farmer devastated after entire crop submerged, needs urgent compensation",
        "My house in {loc} completely destroyed in earthquake, living under tarpaulin now",
        "Small business owner in {loc} financially ruined after cyclone, needs relief",
        "Student from {loc} unable to appear for board exams due to flood displacement",
        "Daily wage worker in {loc} has not earned for {num} days due to disaster situation",
        "Sole breadwinner of family injured in {loc} collapse, wife and children need help",
    ],
}

LOCATIONS = [
    "Assam", "Bihar", "Odisha", "Uttarakhand", "Gujarat", "Kerala",
    "Andhra Pradesh", "Tamil Nadu", "West Bengal", "Himachal Pradesh",
    "Jammu", "Manipur", "Tripura", "Meghalaya", "Guwahati", "Patna",
    "Bhubaneswar", "Dehradun", "Ahmedabad", "Kochi", "Silchar",
    "Dibrugarh", "Kedarnath", "Muzaffarpur", "Srinagar", "Vadodara",
    "Nagpur", "Agartala", "Shillong", "Imphal",
]

ORGS = [
    "NDRF", "Red Cross", "local NGO", "Army", "NDMA",
    "district administration", "State Disaster Authority",
]


def generate(n_per_label_per_event: int = 7, seed: int = 42) -> pd.DataFrame:
    """
    Generates a large realistic disaster tweet dataset.
    n_per_label_per_event: tweets per (label × event) combination
    Default: 7 × 10 labels × 12 events = 840 tweets (fast)
    Set to 20 for ~2400 tweets, 50 for ~6000 tweets
    """
    random.seed(seed)
    rows = []

    for event in EVENTS:
        for label in LABELS:
            templates = TEMPLATES.get(label, TEMPLATES["other_relevant_information"])
            for _ in range(n_per_label_per_event):
                tmpl = random.choice(templates)
                text = tmpl.format(
                    loc=random.choice(LOCATIONS),
                    org=random.choice(ORGS),
                    num=random.randint(2, 500),
                    event=event.replace("_", " "),
                )
                rows.append({
                    "tweet_text":  text,
                    "class_label": label,
                    "event_type":  event,
                    "urgency":     LABEL_TO_URGENCY[label],
                })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def get_or_build(force_rebuild: bool = False, n_per_label: int = 7) -> pd.DataFrame:
    """
    Returns cached dataset if it exists, otherwise builds and saves it.
    force_rebuild=True always regenerates.
    """
    if not force_rebuild and os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)
        print(f"💾 Loaded local dataset from cache: {len(df)} tweets")
        return df

    print(f"🔨 Building local HumAID-style dataset ({n_per_label} samples/label/event)...")
    df = generate(n_per_label_per_event=n_per_label)
    df.to_csv(CACHE_PATH, index=False)
    print(f"✅ Dataset built and saved: {len(df)} tweets → {CACHE_PATH}")
    print(f"   Labels: {df['class_label'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    df = get_or_build(force_rebuild=True, n_per_label=10)
    print(f"\nTotal: {len(df)} tweets")
    print(f"Events: {df['event_type'].nunique()}")
    print(f"\nSample tweets:")
    for _, row in df.sample(5).iterrows():
        print(f"  [{row['urgency']:15}] {row['tweet_text']}")
