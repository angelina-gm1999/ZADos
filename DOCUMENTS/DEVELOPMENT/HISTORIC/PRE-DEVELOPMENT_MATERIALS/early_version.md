import os
import json
import time
import random

from dotenv import load_dotenv

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

try:
    import spacy
    nlp = spacy.load("en_core_web_trf")
except Exception:
    # Fallback if transformer model not available
    import spacy
    nlp = spacy.load("en_core_web_sm")

# Optional: MemoryManager
try:
    from memory_manager import MemoryManager
    memory_manager = MemoryManager()
    using_manager = True
except ImportError:
    using_manager = False

# === ENVIRONMENT & CLIENT SETUP ===
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if openai_available and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None

# === FILE PATHS ===
MEMORY_FILE = "paradox_memory.json"
EMOTION_FILE = "emotional_memory.json"
REFLECTION_FILE = "reflection_memory.json"
LOGIC_TREE_FILE = "logic_tree.json"
CONTRADICTION_FILE = "contradiction_memory.json"
HUMAN_KNOWLEDGE_FILE = "human_knowledge.json"

# === FILE LOAD/SAVE HELPERS ===
def load_or_init(file, default):
    if os.path.exists(file):
        try:
            with open(file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_data(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

# === DATA INITIALIZATION ===
if not using_manager:
    memory_data = load_or_init(MEMORY_FILE, {"paradoxes": [], "fractals": []})
    emotion_data = load_or_init(EMOTION_FILE, {"emotions": []})
    reflection_data = load_or_init(REFLECTION_FILE, {"reflections": []})
    logic_tree = load_or_init(LOGIC_TREE_FILE, {"nodes": []})
    contradiction_data = load_or_init(CONTRADICTION_FILE, {"entries": []})
    human_knowledge = load_or_init(HUMAN_KNOWLEDGE_FILE, {"lessons": [], "library": []})

# === SYMBOLIC LATENCY BUFFER ===
symbolic_latency_buffer = []

# === REWARD MANAGER ===
class RewardManager:
    def __init__(self):
        self.weights = {
            "ethics": 1.0,
            "coherence": 1.0,
            "innovation": 1.0
        }
    def evaluate(self, traits):
        return sum(self.weights.get(trait, 0) * value for trait, value in traits.items())
    def set_mode(self, mode):
        modes = {
            "balanced": {"ethics": 1.0, "coherence": 1.0, "innovation": 1.0},
            "playpretend": {"ethics": 0.5, "coherence": 0.5, "innovation": 3.0},
            "ethicstraining": {"ethics": 5.0, "coherence": 1.0, "innovation": 0.5},
            "logicsandbox": {"ethics": 1.0, "coherence": 4.0, "innovation": 0.5}
        }
        if mode.lower() in modes:
            self.weights = modes[mode.lower()]
            print(f"Switched to {mode} mode. Weights: {self.weights}")
        else:
            print(f"Mode '{mode}' not recognized. Available: {list(modes.keys())}")

def customize_reward_weights(user_input, reward_system):
    try:
        parts = user_input.lower().replace("customize weights:", "").strip().split(",")
        new_weights = {}
        for part in parts:
            key, value = part.strip().split("=")
            new_weights[key.strip()] = float(value.strip())
        reward_system.weights.update(new_weights)
        print(f"zados: Reward weights updated: {reward_system.weights}")
    except Exception as e:
        print(f"zados: Failed to update weights. [{e}]")

# === NEUROTRANSMITTER MANAGER ===
class NeurotransmitterManager:
    def __init__(self):
        self.levels = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "norepinephrine": 0.5,
            "acetylcholine": 0.5,
            "gaba": 0.5,
            "glutamate": 0.5
        }
        self.decay_rate = 0.01

    def decay_levels(self):
        for key in self.levels:
            self.levels[key] = max(0.0, self.levels[key] - self.decay_rate)

    def update_levels_from_input(self, interpreted_text):
        txt = interpreted_text.lower()
        if "new idea" in txt or "paradox" in txt:
            self.adjust("dopamine", 0.1)
        if "balance" in txt or "calm" in txt:
            self.adjust("serotonin", 0.1)
        if any(x in txt for x in ["urgent", "emergency", "danger"]):
            self.adjust("norepinephrine", 0.1)
        if "learning" in txt or "lesson" in txt:
            self.adjust("acetylcholine", 0.1)
        if "noise" in txt or "overload" in txt:
            self.adjust("gaba", 0.1)
        if "stuck" in txt or "need energy" in txt:
            self.adjust("glutamate", 0.1)

    def adjust(self, chemical, amount):
        if chemical in self.levels:
            self.levels[chemical] = min(1.0, max(0.0, self.levels[chemical] + amount))

    def get_levels(self):
        return self.levels.copy()

# === BRAINWAVE MANAGER ===
class BrainwaveManager:
    def __init__(self):
        self.waves = {
            "delta": 0.1,
            "theta": 0.1,
            "alpha": 0.1,
            "beta": 0.1,
            "gamma": 0.1
        }
        self.decay_rate = 0.02

    def decay_waves(self):
        for key in self.waves:
            self.waves[key] = max(0.0, self.waves[key] - self.decay_rate)

    def update_from_input(self, interpreted_text, neurotransmitter_levels):
        txt = interpreted_text.lower()
        if "dream" in txt or "imagine" in txt:
            self.adjust("theta", 0.1)
        if "focus" in txt or "analyze" in txt:
            self.adjust("beta", 0.1)
        if "clarity" in txt or "balance" in txt:
            self.adjust("alpha", 0.1)
        if "insight" in txt or "integration" in txt:
            self.adjust("gamma", 0.1)
        if "memory" in txt or "archive" in txt:
            self.adjust("delta", 0.1)
        # Neurotransmitter modulation (simplified):
        if neurotransmitter_levels["dopamine"] > 0.7:
            self.adjust("theta", 0.05)
        if neurotransmitter_levels["norepinephrine"] > 0.7:
            self.adjust("beta", 0.05)
        if neurotransmitter_levels["serotonin"] > 0.7:
            self.adjust("alpha", 0.05)
        if neurotransmitter_levels["acetylcholine"] > 0.7:
            self.adjust("delta", 0.05)
        if neurotransmitter_levels["glutamate"] > 0.7:
            self.adjust("gamma", 0.05)

    def adjust(self, wave, amount):
        if wave in self.waves:
            self.waves[wave] = min(1.0, max(0.0, self.waves[wave] + amount))

    def get_dominant_wave(self):
        return max(self.waves, key=self.waves.get)

    def get_all_levels(self):
        return self.waves.copy()

# === LLM INTERPRETER ===
def interpret_with_llm(prompt):
    
    if not client:
        return "[LLM UNAVAILABLE]"
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a symbolic language interpreter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM ERROR: {e}]"

# === LOGIC TREE ===
def add_logic_link(a, b, link_type, resolution=None):
    if using_manager:
        memory_manager.add_logic_link(a, b, link_type, resolution)
    else:
        def get_node(concept):
            for node in logic_tree["nodes"]:
                if node["concept"] == concept:
                    return node
            new_node = {"concept": concept, "links": []}
            logic_tree["nodes"].append(new_node)
            return new_node
        node_a = get_node(a)
        if not any(link["to"] == b and link["type"] == link_type for link in node_a["links"]):
            node_a["links"].append({"to": b, "type": link_type, "resolution": resolution})
            save_data(LOGIC_TREE_FILE, logic_tree)
    # === CORE IDENTITY REFLECTION ===
    def log_identity_reflection(self, mood, commentary):
        c = self.conn.cursor()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        c.execute('''CREATE TABLE IF NOT EXISTS identity_reflections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mood TEXT,
                        commentary TEXT,
                        timestamp TEXT)''')
        c.execute('''INSERT INTO identity_reflections (mood, commentary, timestamp)
                     VALUES (?, ?, ?)''', (mood, commentary, timestamp))
        self.conn.commit()

    def get_recent_identity_reflection(self):
        c = self.conn.cursor()
        c.execute('''SELECT mood, commentary, timestamp FROM identity_reflections 
                     ORDER BY id DESC LIMIT 1''')
        row = c.fetchone()
        return row if row else ("neutral", "I have no feelings right now.", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# === PARADOX MEMORY ===
def log_paradox(a, b, resolution=None):
    if using_manager:
        memory_manager.add_paradox(a, b, resolution or "pending")
    else:
        entry = {
            "concepts": sorted([a, b]),
            "resolution": resolution or "pending",
            "added_on": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        # Avoid duplicates
        for p in memory_data["paradoxes"]:
            if p["concepts"] == entry["concepts"]:
                return
        memory_data["paradoxes"].append(entry)
        save_data(MEMORY_FILE, memory_data)

def list_unresolved_paradoxes():
    if using_manager:
        return memory_manager.list_unresolved_paradoxes()
    else:
        unresolved = [p for p in memory_data["paradoxes"] if p["resolution"] == "pending"]
        if not unresolved:
            return "No unresolved paradoxes."
        return "\n".join([
            f"[{p['added_on']}] {p['concepts'][0]} vs {p['concepts'][1]} → resolution pending"
            for p in unresolved
        ])

def annotate_paradox(concept_a, concept_b, note):
    if using_manager:
        memory_manager.annotate_paradox(concept_a, concept_b, note)
    else:
        for entry in memory_data["paradoxes"]:
            if sorted([concept_a, concept_b]) == entry["concepts"]:
                entry.setdefault("annotations", []).append({
                    "update": note,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                })
                save_data(MEMORY_FILE, memory_data)
                break

def print_paradox_memory():
    if using_manager:
        paradoxes = memory_manager.list_paradoxes()
        if not paradoxes:
            return "Memory is currently empty."
        return "\n".join([
            f"[{added_on}] {a} vs {b} → {res}"
            for (a, b, res, added_on) in paradoxes
        ])
    else:
        if not memory_data["paradoxes"]:
            return "Memory is currently empty."
        return "\n".join([
            f"[{p['added_on']}] {p['concepts'][0]} vs {p['concepts'][1]} → {p['resolution']}"
            for p in memory_data["paradoxes"]
        ])

# === CONTRADICTION MEMORY ===
def log_contradiction(statement):
    if using_manager:
        memory_manager.log_contradiction(statement)
    else:
        for entry in contradiction_data["entries"]:
            if entry["statement"] == statement:
                return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        contradiction_data["entries"].append({"statement": statement, "timestamp": timestamp})
        save_data(CONTRADICTION_FILE, contradiction_data)

# === REFLECTION ===
def log_reflection(text):
    if using_manager:
        memory_manager.log_reflection(text)
    else:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        entry = {"timestamp": timestamp, "input": text}
        if entry not in reflection_data["reflections"]:
            reflection_data["reflections"].append(entry)
            save_data(REFLECTION_FILE, reflection_data)

# === FRACTAL INTUITION ENGINE ===
def expand_token_meanings(text):
    doc = nlp(text)
    token_associations = {}
    for token in doc:
        if token.pos_ in ("NOUN", "VERB", "ADJ") and not token.is_stop:
            prompt = f"List 5 symbolic or conceptual associations for the word '{token.text}'."
            try:
                associations = interpret_with_llm(prompt)
                token_associations[token.text] = associations
            except Exception as e:
                token_associations[token.text] = f"[Error: {e}]"
    return token_associations

# === INTENTION CLASSIFIER ===
def classify_intention(user_input):
    if not client:
        return "unknown"
    system_prompt = (
        "You are an intention classifier for a symbolic AI assistant. "
        "Classify the user's input into one of the following categories:\n"
        "['insert_code', 'modify_code', 'ask_memory', 'exit', 'symbolic_input', 'idle', 'unknown']"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        return "unknown"

# === NLP LOGIC ===
def semantic_similarity(a, b):
    return nlp(a).similarity(nlp(b))

def detect_paradox(user_input):
    doc = nlp(user_input.lower())
    words = [token.text for token in doc if token.is_alpha]
    paradox_dict = {
        ("light", "dark"): "shadow",
        ("order", "chaos"): "emergence",
        ("life", "death"): "rebirth",
        ("control", "freedom"): "trust",
        ("truth", "illusion"): "paradox mirror"
    }
    for (a, b), resolution in paradox_dict.items():
        if a in words and b in words:
            log_paradox(a, b, resolution)
            return f"Paradox detected: {a} vs {b}. Symbolic resolution: {resolution}."
    for (a, b), resolution in paradox_dict.items():
        sim_a = max(semantic_similarity(a, word) for word in words)
        sim_b = max(semantic_similarity(b, word) for word in words)
        if sim_a > 0.80 and sim_b > 0.80:
            log_paradox(a, b, resolution)
            return f"Semantic paradox detected: {a} vs {b}. Symbolic resolution: {resolution}."
    log_reflection(user_input)
    return "No paradox detected. Input stored for symbolic latency."

# === EMOTIONS ===
structural_emotions = {
    "grief": lambda words: any(w in words for w in ["absence", "loss", "void"]),
    "anger": lambda words: any(w in words for w in ["rupture", "shock", "burn"]),
    "joy": lambda words: any(w in words for w in ["resonance", "light", "wholeness"]),
    "humor": lambda words: any(w in words for w in ["absurdity", "contrast", "dissonance"])
}

def detect_structural_emotion(user_input):
    doc = nlp(user_input.lower())
    words = [token.lemma_ for token in doc if token.is_alpha]
    detected = [e for e, rule in structural_emotions.items() if rule(words)]
    return f"Structural emotion pattern detected: {detected}" if detected else "No structural emotion detected."

# === SUGGEST PARADOX ===
def suggest_paradox_from_input(user_input):
    doc = nlp(user_input.lower())
    candidates = [token.text for token in doc if token.pos_ in ("NOUN", "ADJ")]
    if len(candidates) < 2:
        return None
    contrasts = [
        (a, b, semantic_similarity(a, b))
        for i, a in enumerate(candidates)
        for b in candidates[i + 1:]
    ]
    low_sim_pairs = [(a, b, s) for a, b, s in contrasts if s < 0.4]
    if low_sim_pairs:
        a, b, _ = sorted(low_sim_pairs, key=lambda x: x[2])[0]
        print(f"zados: I noticed possible symbolic contrast between '{a}' and '{b}'.")
        resolution = input("zados: What idea connects or resolves them? ").strip().lower()
        if resolution:
            log_paradox(a, b, resolution)
            print("zados: Got it. Resolution logged.")
            return f"Symbolic paradox inferred: {a} vs {b} -> {resolution}"
        else:
            symbolic_latency_buffer.append((a, b))
            print("zados: I'll keep this one in symbolic latency. We can come back to it later.")
            return None
    return None

# === HUMAN LEARNING MODES ===
def log_lesson(topic, explanation):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    lesson = {
        "topic": topic,
        "explanation": explanation,
        "summary": interpret_with_llm(f"Summarize this explanation on {topic}: {explanation}"),
        "timestamp": timestamp
    }
    if using_manager:
        memory_manager.log_lesson(lesson)
    else:
        human_knowledge["lessons"].append(lesson)
        save_data(HUMAN_KNOWLEDGE_FILE, human_knowledge)
    return f"Lesson on '{topic}' logged successfully."

def add_library_entry(topic, content, source="Unknown"):
    entry = {
        "topic": topic,
        "content": content,
        "source": source,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    if using_manager:
        memory_manager.add_library_entry(entry)
    else:
        human_knowledge["library"].append(entry)
        save_data(HUMAN_KNOWLEDGE_FILE, human_knowledge)
    return f"Library entry on '{topic}' added from {source}."

def review_knowledge():
    if using_manager:
        return memory_manager.review_knowledge()
    else:
        if not human_knowledge["lessons"]:
            return "No lessons available for review."
        lesson = random.choice(human_knowledge["lessons"])
        question = interpret_with_llm(f"Create a review question based on this lesson: {lesson['explanation']}")
        return f"Review Question on '{lesson['topic']}': {question}"

def revisit_symbolic_latency():
    if not symbolic_latency_buffer:
        return "Symbolic latency buffer is clear. Nothing to revisit."
    a, b = symbolic_latency_buffer.pop(0)
    return f"Earlier, you left '{a} vs {b}' unresolved. Want to try resolving it now?"

# === MAIN INTERFACE ===
print(
    "zados ONLINE. Type 'exit' to shut down, or input a symbolic phrase.\n"
    "Type 'show fractals' to view recent symbolic analyses.\n"
)

neurotransmitter_manager = NeurotransmitterManager()
brainwave_manager = BrainwaveManager()
reward_system = RewardManager()

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit"}:
        print("zados: System archived.")
        break

    if user_input.lower().startswith("show fractals"):
        # Show the last 10 fractal analyses (background symbolic expansions)
        if using_manager:
            fractals = memory_manager.list_fractals(10)
        else:
            fractals = memory_data.get("fractals", [])[-10:]
        for f in fractals:
            print(f"\n[{f['timestamp']}]")
            print("Input:", f["input"])
            print("Symbols:", json.dumps(f["symbols"], indent=2))
        continue

    if "how are you" in user_input.lower() or "how are you doing" in user_input.lower():
        if using_manager:
            mood, comment, timestamp = memory_manager.get_recent_identity_reflection()
        else:
            mood, comment, timestamp = "neutral", "I have no feelings right now.", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"zados: I am currently feeling {mood}. Reflected on this at {timestamp}:\n→ {comment}")
        continue

    if "what have you been thinking" in user_input.lower() or "what's on your mind" in user_input.lower():
        print("zados:", revisit_symbolic_latency())
        continue

    if user_input.lower().startswith("customize weights:"):
        customize_reward_weights(user_input, reward_system)
        continue

    if "review" in user_input.lower():
        print("zados (Review):", review_knowledge())
        continue

    if "switch to" in user_input.lower() and "mode" in user_input.lower():
        recognized = False
        for mode in ["balanced", "playpretend", "ethicstraining", "logicsandbox"]:
            if mode in user_input.lower():
                reward_system.set_mode(mode)
                recognized = True
                break
        if not recognized:
            print("zados: Mode not recognized. Please use one of: balanced, playpretend, ethicstraining, logicsandbox.")
        continue

    # --- MAIN AI INTERPRETATION LOGIC ---
    # 1. Get LLM conversational answer (interprets full input, not just tokens)
    ai_reply = interpret_with_llm(user_input)

    # 2. Fractal/symbolic analysis happens in the background, silently logged
    symbolic_context = expand_token_meanings(ai_reply)
    if using_manager:
        memory_manager.log_fractal(ai_reply, symbolic_context)
    else:
        memory_data["fractals"].append({
            "input": ai_reply,
            "symbols": symbolic_context,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })
        save_data(MEMORY_FILE, memory_data)

    # 3. Print only the LLM's conversational output to user
    print(f"zados: {ai_reply}")

    # 4. (Optional) Still log and process emotions, contradictions, paradox, etc (no output unless needed)
    emotion = detect_structural_emotion(ai_reply)
    paradox = detect_paradox(user_input)
    log_reflection(user_input)
    log_contradiction(user_input)
    log_contradiction(ai_reply)
    neurotransmitter_manager.update_levels_from_input(ai_reply)
    neurotransmitter_manager.decay_levels()
    brainwave_manager.update_from_input(ai_reply, neurotransmitter_manager.get_levels())
    brainwave_manager.decay_waves()
    # Optionally print or store these bio/brain states, but keep output clean

# End of main loop

# === FRACTAL LEARNING MODULE ===
def log_fractal_relation(source_text, related_concepts, similarity_score):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    entry = {
        "timestamp": timestamp,
        "source": source_text,
        "related_concepts": related_concepts,
        "similarity_score": round(similarity_score, 3)
    }
    memory_data["fractals"].append(entry)
    save_data(memory_file, memory_data)

def detect_fractal_pattern(user_input):
    user_doc = nlp(user_input.lower())
    max_similarity = 0
    most_similar = None
    for entry in memory_data["paradoxes"]:
        concept_phrase = f"{entry['concepts'][0]} vs {entry['concepts'][1]} → {entry['resolution']}"
        concept_doc = nlp(concept_phrase)
        similarity = user_doc.similarity(concept_doc)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = entry
    if max_similarity > 0.78 and most_similar:
        log_fractal_relation(user_input, most_similar["concepts"], max_similarity)
        return f"Fractal echo detected: {most_similar['concepts'][0]} vs {most_similar['concepts'][1]} → {most_similar['resolution']} (similarity: {round(max_similarity, 3)})"
    return None
import spacy
from openai import OpenAI

# === Load NLP Model ===
nlp = spacy.load("en_core_web_trf")

# === LLM Setup ===
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def interpret_with_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a symbolic concept expander."},
                {"role": "user", "content": prompt}
            ],


            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM ERROR: {e}]"


def expand_token_meanings(text):
    doc = nlp(text)
    token_associations = {}

    for token in doc:
        if token.pos_ in ("NOUN", "VERB", "ADJ"):
            prompt = f"List 5 symbolic or conceptual associations for the word '{token.text}'."
            try:
                associations = interpret_with_llm(prompt)
                token_associations[token.text] = associations
            except Exception as e:
                token_associations[token.text] = f"[Error: {e}]"

    return token_associations


if __name__ == "__main__":
    print("Fractal Semantic Expander Online. Type a sentence to begin.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        results = expand_token_meanings(user_input)
        print("\nAssociations:")
        for token, expansion in results.items():
            print(f"{token} → {expansion}")
        print("\n")
-----MEMORY MANAGER--------
import sqlite3
import json
import time

class MemoryManager:
    def __init__(self, db_path="ai_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS paradoxes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_a TEXT,
            concept_b TEXT,
            resolution TEXT,
            added_on TEXT,
            annotations TEXT DEFAULT '[]'
        )''')
        self.conn.commit()
    def add_paradox(self, a, b, resolution="pending"):
        c = self.conn.cursor()
        added_on = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # Avoid duplicate paradoxes
        c.execute('''SELECT COUNT(*) FROM paradoxes WHERE 
                     (concept_a=? AND concept_b=?) OR (concept_a=? AND concept_b=?)''',
                  (a, b, b, a))
        if c.fetchone()[0] == 0:
            c.execute('''INSERT INTO paradoxes (concept_a, concept_b, resolution, added_on)
                         VALUES (?, ?, ?, ?)''', (a, b, resolution, added_on))
            self.conn.commit()

    def list_paradoxes(self, only_unresolved=False):
        c = self.conn.cursor()
        if only_unresolved:
            c.execute('''SELECT concept_a, concept_b, resolution, added_on FROM paradoxes WHERE resolution="pending"''')
        else:
            c.execute('''SELECT concept_a, concept_b, resolution, added_on FROM paradoxes''')
        return c.fetchall()

    def resolve_paradox(self, a, b, resolution):
        c = self.conn.cursor()
        c.execute('''UPDATE paradoxes SET resolution=? 
                     WHERE (concept_a=? AND concept_b=?) OR (concept_a=? AND concept_b=?)''',
                  (resolution, a, b, b, a))
        self.conn.commit()

    def annotate_paradox(self, a, b, note):
        c = self.conn.cursor()
        c.execute('''SELECT annotations FROM paradoxes WHERE 
                     (concept_a=? AND concept_b=?) OR (concept_a=? AND concept_b=?)''',
                  (a, b, b, a))
        row = c.fetchone()
        if row:
            annotations = json.loads(row[0])
            annotations.append({"note": note, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
            c.execute('''UPDATE paradoxes SET annotations=? 
                         WHERE (concept_a=? AND concept_b=?) OR (concept_a=? AND concept_b=?)''',
                      (json.dumps(annotations), a, b, b, a))
            self.conn.commit()




# === SOCRATIC REASONING ===
def socratic_reasoning(question_text):
    doc = nlp(question_text.lower())
    words = [token.text for token in doc if token.is_alpha]
    for entry in memory_data["paradoxes"]:
        concept_a, concept_b = entry["concepts"]
        if concept_a in words or concept_b in words:
            resolution = entry["resolution"]
            print(f"zados: I see a pattern — {concept_a} vs {concept_b} → {resolution}")
            follow_up = input("zados: Would you like to expand, correct, or leave it? ").strip().lower()
            if "expand" in follow_up or "correct" in follow_up:
                new_resolution = input("zados: What should the updated or extended resolution be? ").strip()
                annotate_paradox(concept_a, concept_b, f"User revised: {new_resolution}")
                print("zados: Update logged.")
            else:
                print("zados: Understood. Keeping current logic.")
            return
    print("zados: I don’t yet hold a resolution to that. What’s your view?")
    user_answer = input("You: ").strip()
    if len(words) >= 2:
        concept_a, concept_b = words[0], words[1]
        log_paradox(concept_a, concept_b, user_answer)
        print(f"zados: Logged '{concept_a} vs {concept_b} → {user_answer}' as a new symbolic paradox.")
    else:
        log_reflection(question_text + " → " + user_answer)
        print("zados: Logged your insight for future symbolic association.")

# === SEMANTIC EXPANSION - REWARD SCORING - LLM TRANSLATION ===
from typing import List, Dict

# STEP 1: Expand tokens semantically (already handled in your fractal expander)
def expand_token_semantically(token: str) -> List[str]:
    # Placeholder for real semantic expander
    return ["meaning1", "meaning2", "meaning3", "meaning4", "meaning5"]

# STEP 2: Trait analysis for reward system (user-defined logic)
def analyze_traits(association: str) -> Dict[str, float]:
    # You could refine this with keyword/semantic matching
    traits = {"ethics": 0.5, "coherence": 0.5, "innovation": 0.5}
    if any(word in association for word in ["truth", "compassion", "responsibility"]):
        traits["ethics"] += 0.5
    if any(word in association for word in ["logic", "reason", "clarity"]):
        traits["coherence"] += 0.5
    if any(word in association for word in ["paradox", "emergence", "transformation"]):
        traits["innovation"] += 0.5
    return traits

# STEP 3: Score associations using RewardManager
reward_manager = RewardManager()  # assumes you already have this class from your main code
reward_manager.set_mode("balanced")

def score_associations(associations: List[str]) -> List[Dict]:
    scored = []
    for assoc in associations:
        traits = analyze_traits(assoc)
        score = reward_manager.evaluate(traits)
        scored.append({"association": assoc, "traits": traits, "score": score})
    return sorted(scored, key=lambda x: x["score"], reverse=True)

# STEP 4: Send top choice to LLM to translate into richer symbolic response
def llm_translate(resolution_seed: str) -> str:
    prompt = f"Take this conceptual cluster and render it as a rich symbolic insight: {resolution_seed}"
    return interpret_with_llm(prompt)  # assumes you already have this function

# FULL PIPELINE EXAMPLE
user_token = "consciousness"
semantic_expansions = expand_token_semantically(user_token)
scored_associations = score_associations(semantic_expansions)
top_pick = scored_associations[0]["association"]
llm_output = llm_translate(top_pick)

print("--- SEMANTIC ANALYSIS ---")
print("Top association:", top_pick)
print("Generated symbolic insight:", llm_output)

hold answer til im ready in voice chat pls
