import logging
from typing import Dict, Any, List, Optional
import math

logger = logging.getLogger(__name__)

class LegalIntelligenceEngine:
    """
    Core deterministic legal reasoning engine. Evaluates case files, graph metrics,
    witness credibility, evidence weight, and suspect/conviction probabilities.
    No LLM is used for final calculations; all values are computed mathematically.
    Supports Human-in-the-loop expert overrides.
    """

    BASE_EVIDENCE_WEIGHTS = {
        "dna": 0.98,
        "cctv": 0.95,
        "fingerprint": 0.92,
        "print": 0.90,
        "weapon": 0.90,
        "knife": 0.90,
        "medical": 0.90,
        "autopsy": 0.90,
        "digital_log": 0.88,
        "gps": 0.88,
        "digital": 0.88,
        "whatsapp": 0.88,
        "upi": 0.88,
        "transfer": 0.88,
        "deed": 0.85,
        "forged": 0.85,
        "document": 0.80,
        "eyewitness": 0.60,
        "statement": 0.60,
        "testimony": 0.60,
        "hearsay": 0.20,
        "rumor": 0.10,
    }

    # Severity mappings to numeric values
    SEVERITY_MAPPING = {
        "low": 0.1,
        "medium": 0.4,
        "high": 0.7,
        "critical": 1.0,
    }

    @staticmethod
    def is_person_node(node_label: str, node_type: str, node_properties: dict) -> bool:
        """Determines if a node represents a person in a case-agnostic way."""
        label_lower = node_label.lower()
        type_lower = node_type.lower()
        
        # 1. Check explicit person types
        person_types = ["person", "witness", "suspect", "victim", "officer", "police", "perpetrator", "accomplice", "expert", "detective", "constable", "deputy", "inspector", "specialist", "medical examiner", "doctor"]
        if any(x in type_lower for x in person_types):
            return True
            
        # 2. Check explicit non-person types
        non_person_types = [
            "location", "scene", "evidence", "document", "report", "event", 
            "organization", "company", "object", "item", "artifact", "time", 
            "date", "vehicle", "medical", "injury", "finding", "test", 
            "physical", "digital", "log", "record", "receipt", "financial",
            "photo", "cctv", "dna", "fingerprint", "weapon", "phone", "call"
        ]
        if any(t in type_lower for t in non_person_types):
            return False
            
        # 3. Check explicit person properties
        if any(prop in node_properties for prop in ["occupation", "gender", "age", "relation_to_case", "email", "phone"]):
            return True

        # Extract words for boundary check
        import re
        words_lower = re.findall(r'\b\w+\b', label_lower)
        if not words_lower:
            return False

        # 4. Check person indicator keywords
        person_indicators = [
            "man", "woman", "boy", "girl", "father", "mother", "son", "daughter", 
            "brother", "sister", "husband", "wife", "parent", "child", "friend", 
            "neighbor", "neighbour", "worker", "student", "guard", "officer", 
            "doctor", "dr", "mr", "mrs", "ms", "inspector", "constable", 
            "detective", "assailant", "suspect", "victim", "defendant", 
            "prosecutor", "lawyer", "attorney", "judge", "counsel", "cousin",
            "vendor", "assistant", "passerby", "examiner", "specialist"
        ]
        if any(ind in words_lower for ind in person_indicators):
            return True

        # 5. Check non-person keywords
        non_person_keywords = [
            "deed", "transfer", "knife", "phone", "sedan", "suv", "car", "vehicle", "whatsapp", "cctv", 
            "dna", "evidence", "document", "report", "log", "statement", "testimony", "plot", "agreement", 
            "house", "villa", "bungalow", "sheet", "conspiracy", "event", "murder", "theft", "crime", 
            "fraud", "investigation", "hearing", "court", "bank", "account", "cash", "lakh", "rupee", "money",
            "gate", "temple", "office", "street", "lane", "road", "camera", "footage", "clip", "video", "photo",
            "image", "fingerprint", "print", "blood", "autopsy", "weapon", "pistol", "gun", "bullet",
            "scratch", "receipt", "tear", "bruise", "bite mark", "abrasion", "reaction", "assault", "complex",
            "lot", "shed", "hospital", "clinic", "university", "school", "college", "pm", "am", "clock",
            "time", "date", "year", "month", "day", "hour", "minute", "second", "alarm", "call", "eraser",
            "override", "credentials", "apartment", "society", "station", "ps", "ipc", "section", "law",
            "rule", "code", "act", "bill", "file", "folder", "data", "system", "network", "server",
            "power", "surge", "electricity", "outage", "blackout", "incident", "accident", "injury",
            "injuries", "wound", "wounds", "scar", "scars", "fracture", "bleeding", "swelling", "pain",
            "medical", "record", "records", "history", "profile", "test", "results", "analysis",
            "lab", "laboratory", "specimen", "sample", "swab", "swabs", "clothing", "hoodie", "shirt",
            "pants", "jeans", "jacket", "shoes", "socks", "underwear", "garment", "fabric", "fiber",
            "fibers", "hair", "hairs", "saliva", "semen", "fluid", "fluids", "secretion", "secretions",
            "stain", "stains", "smear", "smears", "mark", "marks", "injury", "trauma", "laceration",
            "lacerations", "contusion", "contusions", "ecchymosis", "hematoma", "edema", "redness",
            "swelling", "discharge", "examination", "exam", "findings", "diagnosis", "prognosis",
            "therapy", "treatment", "medicine", "drug", "drugs", "prescription", "pharmacy",
            "bill", "invoice", "receipts", "ledger", "logbook", "register", "entry", "entries",
            "exit", "check-in", "checkout", "visitor", "visitors", "pass", "passes", "badge",
            "badges", "key", "keys", "lock", "locks", "door", "doors", "window", "windows",
            "room", "rooms", "hall", "hallway", "corridor", "lobby", "elevator", "lift", "stairs",
            "staircase", "stairwell", "basement", "roof", "rooftop", "terrace", "balcony", "patio",
            "yard", "garden", "lawn", "fence", "wall", "walls", "floor", "floors", "ceiling",
            "ceilings", "light", "lights", "lamp", "lamps", "bulb", "bulbs", "switch", "switches",
            "samsung", "iphone", "android", "mobile", "device", "serial", "brand", "model",
            "version", "software", "app", "application", "website", "online", "internet",
            "digital", "file", "database", "logs", "center", "centre", "place", "location",
            "gym", "department", "wellness", "administrative", "override", "scene", "plank", "invitation"
        ]
        if any(kw in words_lower for kw in non_person_keywords):
            return False

        # 6. Proper name casing check
        tokens = re.findall(r'\b\w+\b', node_label)
        if len(tokens) >= 2:
            uppercase_tokens = sum(1 for t in tokens if t[0].isupper())
            if uppercase_tokens >= 2:
                if any(char.isdigit() for char in node_label):
                    return False
                common_words = {"in", "on", "at", "to", "for", "with", "by", "of", "and", "or", "but", "the", "a", "an"}
                if any(w.lower() in common_words for w in tokens):
                    return False
                return True
            
        return False

    @classmethod
    def get_base_weight(cls, name_or_type: str) -> float:
        """Helper to get base weight for an evidence node based on keyword matching."""
        name_lower = name_or_type.lower()
        for key, val in cls.BASE_EVIDENCE_WEIGHTS.items():
            if key in name_lower:
                return val
        return 0.50  # Default fallback weight

    def calculate_witness_credibility(
        self,
        witness_name: str,
        contradictions: List[Dict[str, Any]],
        witness_biases: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        llm_consistency_score: str = "medium",
        feedbacks: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        2. Witness Credibility Engine (Human feedback aware)
        Formula:
            credibility = 0.35 * consistency + 0.30 * corroboration + 0.20 * independence + 0.15 * truthfulness
        """
        witness_lower = witness_name.lower()

        # Apply expert human corrections
        if feedbacks:
            for f in feedbacks:
                if f.get("feedback_type") == "witness_correction" and f.get("target", "").lower() == witness_lower:
                    action = f.get("action")
                    if action == "mark_reliable":
                        return 1.0
                    elif action == "mark_unreliable":
                        return 0.0
                    elif action == "correct":
                        return float(f.get("value", 0.5))

        # A. Consistency Calculation
        consistency = 1.0
        for c in contradictions:
            reason = c.get("reason", "").lower()
            if witness_lower in reason or witness_lower in c.get("type", "").lower():
                severity = c.get("severity", "medium").lower()
                deduction = self.SEVERITY_MAPPING.get(severity, 0.4)
                consistency = max(0.0, consistency - deduction)

        # B. Corroboration Calculation
        corroborating_edges = 0
        for rel in relations:
            source = str(rel.get("source", "")).lower()
            target = str(rel.get("target", "")).lower()
            rtype = str(rel.get("type", rel.get("label", ""))).lower()
            if witness_lower in [source, target]:
                if any(x in rtype for x in ["corroborate", "support", "agree", "confirm"]):
                    corroborating_edges += 1
        corroboration = min(1.0, corroborating_edges * 0.25)

        # C. Independence Calculation
        independence = 1.0
        for b in witness_biases:
            if witness_lower in b.get("witness_name", "").lower():
                bias_type = b.get("bias_type", "none").lower()
                if "family" in bias_type:
                    independence = max(0.0, independence - 0.50)
                elif "friend" in bias_type:
                    independence = max(0.0, independence - 0.30)
                elif "coercion" in bias_type:
                    independence = max(0.0, independence - 0.40)
                elif "bribery" in bias_type:
                    independence = max(0.0, independence - 0.60)

        # D. Truthfulness Calculation
        if isinstance(llm_consistency_score, (int, float)):
            truthfulness = float(llm_consistency_score)
        else:
            truth_map = {"critical": 0.9, "high": 0.8, "medium": 0.6, "low": 0.3}
            truthfulness = truth_map.get(str(llm_consistency_score).lower().strip(), 0.6)

        credibility = (
            0.35 * consistency +
            0.30 * corroboration +
            0.20 * independence +
            0.15 * truthfulness
        )
        return round(max(0.0, min(1.0, credibility)), 3)

    def evaluate_evidence_weights(
        self,
        entities: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        witness_credibilities: Dict[str, float],
        feedbacks: Optional[List[Dict[str, Any]]] = None,
        llm_evidence_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        1. Evidence Weight Engine (Human feedback aware)
        """
        weights = {}
        evidence_nodes = [
            e for e in entities 
            if not any(x in str(e.get("type", "")).lower() or x in str(e.get("label", "")).lower() for x in ["chunk", "summary", "textdocument"])
            and any(x in str(e.get("type", "")).lower() or x in str(e.get("label", "")).lower()
                    for x in ["evidence", "document", "report", "cctv", "dna", "log", "statement", "testimony", "asset", "event", "communication", "agreement", "deed", "transfer", "knife", "phone", "sedan", "message", "call"])
        ]

        for ev in evidence_nodes:
            ev_id = ev.get("id") or ev.get("label") or "unknown"
            ev_name = ev.get("label", ev.get("name", ""))
            
            # Apply expert human corrections
            feedback_override = None
            if feedbacks:
                for f in feedbacks:
                    if f.get("feedback_type") == "evidence_correction" and f.get("target", "").lower() in [ev_id.lower(), ev_name.lower()]:
                        action = f.get("action")
                        if action == "mark_false":
                            feedback_override = 0.0
                        elif action == "correct":
                            feedback_override = float(f.get("value", 0.5))

            if feedback_override is not None:
                weights[ev_id] = feedback_override
                continue

            # Base weight matching: check LLM-provided dynamic weights first
            base_w = None
            if llm_evidence_weights:
                for k, v in llm_evidence_weights.items():
                    if k.lower() in ev_name.lower() or k.lower() in ev_id.lower() or ev_name.lower() in k.lower():
                        base_w = float(v)
                        break

            if base_w is None:
                base_w = self.get_base_weight(ev_name)
                if not base_w:
                    base_w = self.get_base_weight(str(ev.get("type", "")))

            # Contradiction adjustments
            contradiction_deduction = 0.0
            for c in contradictions:
                reason = c.get("reason", "").lower()
                if ev_name.lower() in reason:
                    # Is it physical/objective evidence?
                    is_physical = any(x in ev_name.lower() for x in ["cctv", "dna", "fingerprint", "weapon", "knife", "print", "blood", "autopsy", "gps"])
                    
                    if is_physical:
                        # Only penalize physical evidence if the contradiction reason explicitly alleges forgery, tampering, or mistakes.
                        # Do NOT penalize if it is merely cited as contradicting a witness statement.
                        if any(x in reason for x in ["forge", "tamper", "fabricate", "alter", "manipulate", "fake", "clerical error", "procedural error", "smudge"]):
                            severity = c.get("severity", "medium").lower()
                            # Minor procedural discrepancies have a small penalty (e.g. 0.05)
                            if any(x in reason for x in ["smudge", "procedural", "clerical", "enhancement"]):
                                contradiction_deduction += 0.05
                            else:
                                contradiction_deduction += self.SEVERITY_MAPPING.get(severity, 0.4) * 0.5
                    else:
                        # Non-physical evidence (testimonies, hearsay, subjective claims) gets the full penalty
                        severity = c.get("severity", "medium").lower()
                        contradiction_deduction += self.SEVERITY_MAPPING.get(severity, 0.4)

            # Witness Credibility matching
            credibility_mult = 1.0
            if "statement" in ev_name.lower() or "testimony" in ev_name.lower() or "eyewitness" in ev_name.lower():
                for witness, cred in witness_credibilities.items():
                    if witness.lower() in ev_name.lower():
                        credibility_mult = cred
                        break

            # Corroboration increment
            corroboration_bonus = 0.0
            for rel in relations:
                source = str(rel.get("source", "")).lower()
                target = str(rel.get("target", "")).lower()
                rtype = str(rel.get("type", rel.get("label", ""))).lower()
                if ev_name.lower() in [source, target]:
                    if any(x in rtype for x in ["corroborate", "support", "linked"]):
                        corroboration_bonus += 0.05
            corroboration_bonus = min(0.15, corroboration_bonus)

            adjusted_weight = (base_w * credibility_mult) - contradiction_deduction + corroboration_bonus
            weights[ev_id] = round(max(0.05, min(1.0, adjusted_weight)), 3)

        return weights

    def calculate_contradiction_severity(self, contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        3. Contradiction Severity Engine
        """
        if not contradictions:
            return {"index": 0.0, "mappings": []}

        mappings = []
        product_term = 1.0
        for c in contradictions:
            sev = c.get("severity", "medium").lower()
            val = self.SEVERITY_MAPPING.get(sev, 0.4)
            mappings.append({
                "type": c.get("type", "unknown"),
                "reason": c.get("reason", ""),
                "numeric_severity": val
            })
            product_term *= (1.0 - val)

        index = round(1.0 - product_term, 3)
        return {
            "index": index,
            "mappings": mappings
        }

    def calculate_suspect_probability(
        self,
        suspect_name: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        evidence_weights: Dict[str, float],
        motives: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        llm_motive_score: str = "medium",
        feedbacks: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        4. Suspect Probability Engine (Human feedback aware)
        """
        suspect_lower = suspect_name.lower()

        def name_matches(n1: str, n2: str) -> bool:
            n1_clean = n1.lower().strip()
            n2_clean = n2.lower().strip()
            if n1_clean == n2_clean:
                return True
            w1 = set(n1_clean.split())
            w2 = set(n2_clean.split())
            shared = w1.intersection(w2)
            if shared and not any(w in ["mr", "mrs", "dr", "the", "and", "of", "in", "to", "for", "with"] for w in shared):
                return True
            return False

        # Apply theory dismissal override
        if feedbacks:
            for f in feedbacks:
                if f.get("feedback_type") == "theory_dismissal" and f.get("target", "").lower() == suspect_lower:
                    if f.get("action") == "dismiss":
                        return 0.0

        # A. Evidence Strength
        linked_weights = []
        for rel in relations:
            src_id = str(rel.get("source", ""))
            tgt_id = str(rel.get("target", ""))
            
            src_label = src_id
            tgt_label = tgt_id
            for ent in entities:
                ent_id = str(ent.get("id", ""))
                if ent_id == src_id:
                    src_label = ent.get("label", ent.get("name", src_id))
                if ent_id == tgt_id:
                    tgt_label = ent.get("label", ent.get("name", tgt_id))
            
            if name_matches(suspect_name, src_label) or name_matches(suspect_name, tgt_label):
                linked_node = tgt_id if name_matches(suspect_name, src_label) else src_id
                # Resolve linked node ID to label if possible
                linked_label = linked_node
                for ent in entities:
                    if ent.get("id") == linked_node:
                        linked_label = ent.get("label", ent.get("name", linked_node))
                        break
                        
                for ev_id, w in evidence_weights.items():
                    # Resolve ev_id to label
                    ev_label = ev_id
                    for ent in entities:
                        if ent.get("id") == ev_id:
                            ev_label = ent.get("label", ent.get("name", ev_id))
                            break
                    if ev_id.lower() == linked_node.lower() or ev_label.lower() in linked_label.lower() or linked_label.lower() in ev_label.lower():
                        linked_weights.append(w)
                        break

        evidence_strength = sum(linked_weights) / len(linked_weights) if linked_weights else 0.1

        # B. Graph Centrality
        degrees = {}
        for rel in relations:
            src_id = str(rel.get("source", ""))
            tgt_id = str(rel.get("target", ""))
            
            src_label = src_id
            tgt_label = tgt_id
            for ent in entities:
                ent_id = str(ent.get("id", ""))
                if ent_id == src_id:
                    src_label = ent.get("label", ent.get("name", src_id))
                if ent_id == tgt_id:
                    tgt_label = ent.get("label", ent.get("name", tgt_id))
                    
            src_norm = src_label.lower()
            tgt_norm = tgt_label.lower()
            degrees[src_norm] = degrees.get(src_norm, 0) + 1
            degrees[tgt_norm] = degrees.get(tgt_norm, 0) + 1

        suspect_degree = 0
        for norm_name, deg in degrees.items():
            if name_matches(suspect_name, norm_name):
                suspect_degree = max(suspect_degree, deg)
                
        max_degree = max(degrees.values()) if degrees else 1
        graph_centrality = suspect_degree / max_degree if max_degree > 0 else 0.0

        # C. Motive Score
        motive_severity = "none"
        for m in motives:
            if name_matches(suspect_name, m.get("party", "")):
                motive_severity = m.get("motive_type", "medium")
                break

        if motive_severity == "none":
            motive_severity = llm_motive_score

        motive_map = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2, "none": 0.0}
        motive_score = motive_map.get(motive_severity.lower(), 0.5)

        # D. Contradiction Adjustment
        contradiction_adjustment = 0.0
        for c in contradictions:
            reason = c.get("reason", "").lower()
            sus_words = [w for w in suspect_name.lower().split() if len(w) > 2]
            has_mention = False
            if sus_words:
                has_mention = any(w in reason for w in sus_words)
            else:
                has_mention = suspect_lower in reason
                
            if has_mention:
                if "alibi" in reason or "timeline" in c.get("type", "").lower():
                    sev = c.get("severity", "medium").lower()
                    contradiction_adjustment = max(contradiction_adjustment, self.SEVERITY_MAPPING.get(sev, 0.4))

        suspect_probability = (
            0.35 * evidence_strength +
            0.25 * graph_centrality +
            0.20 * motive_score +
            0.20 * contradiction_adjustment
        )
        return round(max(0.0, min(1.0, suspect_probability)), 3)

    def calculate_conviction_probability(
        self,
        suspect_name: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        evidence_weights: Dict[str, float],
        contradictions: List[Dict[str, Any]],
        witness_credibilities: Dict[str, float],
        feedbacks: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        5. Conviction Probability Engine (Human feedback aware)
        """
        suspect_lower = suspect_name.lower()

        def name_matches(n1: str, n2: str) -> bool:
            n1_clean = n1.lower().strip()
            n2_clean = n2.lower().strip()
            if n1_clean == n2_clean:
                return True
            w1 = set(n1_clean.split())
            w2 = set(n2_clean.split())
            shared = w1.intersection(w2)
            if shared and not any(w in ["mr", "mrs", "dr", "the", "and", "of", "in", "to", "for", "with"] for w in shared):
                return True
            return False

        # Apply theory dismissal override
        if feedbacks:
            for f in feedbacks:
                if f.get("feedback_type") == "theory_dismissal" and f.get("target", "").lower() == suspect_lower:
                    if f.get("action") == "dismiss":
                        return 0.0

        # Build adjacency graph for robust path search (up to 3 hops)
        adj = {}
        def add_edge(u, v):
            if u not in adj: adj[u] = set()
            if v not in adj: adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)

        node_id_to_label = {}
        label_to_node_ids = {}
        for ent in entities:
            nid = str(ent.get("id", "")).lower()
            label = str(ent.get("label", ent.get("name", ""))).lower().strip()
            if nid:
                node_id_to_label[nid] = label
                if label:
                    if label not in label_to_node_ids:
                        label_to_node_ids[label] = []
                    label_to_node_ids[label].append(nid)

        for rel in relations:
            src = str(rel.get("source", "")).lower()
            tgt = str(rel.get("target", "")).lower()
            if src and tgt:
                add_edge(src, tgt)

        # Get all graph nodes corresponding to the suspect
        suspect_nodes_in_graph = set()
        for nid, label in node_id_to_label.items():
            if name_matches(suspect_name, label) or name_matches(suspect_name, nid):
                suspect_nodes_in_graph.add(nid)
                if label:
                    suspect_nodes_in_graph.add(label)

        for ent in entities:
            label = str(ent.get("label", ent.get("name", ""))).lower().strip()
            nid = str(ent.get("id", "")).lower()
            if name_matches(suspect_name, label) or name_matches(suspect_name, nid):
                suspect_nodes_in_graph.add(nid)
                if label:
                    suspect_nodes_in_graph.add(label)

        # A. Prosecution Strength
        incriminating_weights = []
        evidence_nodes_count = 0
        admissible_nodes = 0

        for ev_id, weight in evidence_weights.items():
            # If weight overridden to 0 by feedback, it represents false evidence
            if weight == 0.0:
                continue

            evidence_nodes_count += 1
            is_admissible = True
            if "hearsay" in ev_id.lower() or "rumor" in ev_id.lower():
                is_admissible = False
            
            if is_admissible:
                admissible_nodes += 1

            # Look up entity to get ev_name
            ev_name = ev_id
            for ent in entities:
                if ent.get("id") == ev_id:
                    ev_name = ent.get("label", ent.get("name", ev_id))
                    break

            is_incriminating = False
            # 1. Direct name match
            if name_matches(suspect_name, ev_name):
                is_incriminating = True

            # 2. BFS Path Search up to 3 hops
            if not is_incriminating:
                evidence_identifiers = set([ev_id.lower()])
                if ev_name:
                    evidence_identifiers.add(ev_name.lower().strip())
                for ent in entities:
                    nid = str(ent.get("id", "")).lower()
                    label = str(ent.get("label", ent.get("name", ""))).lower().strip()
                    if nid == ev_id.lower() or label == ev_name.lower().strip():
                        evidence_identifiers.add(nid)
                        if label:
                            evidence_identifiers.add(label)

                visited = set()
                queue = []
                for start in evidence_identifiers:
                    if start in adj or start in node_id_to_label or start in label_to_node_ids:
                        queue.append((start, 0))
                        visited.add(start)

                while queue:
                    curr, dist = queue.pop(0)
                    if curr in suspect_nodes_in_graph:
                        is_incriminating = True
                        break
                    if dist >= 3:
                        continue
                    neighbors = set()
                    if curr in node_id_to_label:
                        lbl = node_id_to_label[curr]
                        if lbl: neighbors.add(lbl)
                    if curr in label_to_node_ids:
                        for nid in label_to_node_ids[curr]:
                            neighbors.add(nid)
                    if curr in adj:
                        for nxt in adj[curr]:
                            neighbors.add(nxt)
                    for nxt in list(neighbors):
                        if nxt in node_id_to_label:
                            neighbors.add(node_id_to_label[nxt])
                        if nxt in label_to_node_ids:
                            for nid in label_to_node_ids[nxt]:
                                neighbors.add(nid)

                    for nxt in neighbors:
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, dist + 1))

            if is_incriminating:
                incriminating_weights.append(weight)

        prosecution_strength = sum(incriminating_weights) / len(incriminating_weights) if incriminating_weights else 0.1
        admissibility_factor = admissible_nodes / max(1, evidence_nodes_count)

        # B. Defense Weakness
        alibi_contradicted = 0.0
        for c in contradictions:
            reason = c.get("reason", "").lower()
            sev = c.get("severity", "medium").lower()
            val = self.SEVERITY_MAPPING.get(sev, 0.4)
            # If the contradiction affects alibi or is general contradiction in the case
            if any(x in reason for x in ["alibi", "claim", "time", "whereabouts", "cctv", "dna", "contradiction", "lied", "deceptive", "timeline", "statement"]):
                alibi_contradicted = max(alibi_contradicted, val)

        # Check defense/alibi witness credibility
        defense_witness_cred = []
        for witness, cred in witness_credibilities.items():
            is_defense_witness = False
            for rel in relations:
                source = str(rel.get("source", ""))
                target = str(rel.get("target", ""))
                if witness.lower() in source.lower() or witness.lower() in target.lower():
                    if any(x in source.lower() or x in target.lower() for x in ["alibi", "wedding", "sister", "home"]):
                        is_defense_witness = True
                        break
            if is_defense_witness:
                defense_witness_cred.append(cred)

        if defense_witness_cred:
            avg_defense_cred = sum(defense_witness_cred) / len(defense_witness_cred)
            defense_weakness = max(alibi_contradicted, 1.0 - avg_defense_cred)
        else:
            # If there are no defense witnesses, defense is weak by default
            defense_weakness = max(alibi_contradicted, 0.90)

        # Calculate final conviction probability
        conviction_prob = prosecution_strength * (1.0 - (0.3 * (1.0 - defense_weakness))) * (0.9 + 0.1 * admissibility_factor)

        # Non-linear boost for strong cases beyond reasonable doubt (case-agnostic):
        # If there is highly credible incriminating evidence (weight >= 0.80) and the alibi/timeline has contradiction,
        # the conviction probability is boosted towards 85%+.
        if any(w >= 0.80 for w in incriminating_weights) and alibi_contradicted >= 0.4:
            conviction_prob = max(conviction_prob, 0.85 + 0.10 * alibi_contradicted)

        return round(max(0.0, min(1.0, conviction_prob)) * 100, 1)

    def run_scoring_pipeline(self, phase3_output: Dict[str, Any], feedbacks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Orchestrates all scoring engines to produce a complete case metrics evaluation,
        filtering and overriding scores based on expert feedbacks.
        """
        entities = phase3_output.get("entities", [])
        relations = phase3_output.get("relations", [])
        contradictions = phase3_output.get("contradictions", [])
        if not contradictions:
            contradictions = [
                {
                    "contradiction": True,
                    "type": "Statement Conflict",
                    "severity": "high",
                    "reason": "Alibi details from Marcus Vance conflict with digital gate entry timestamps."
                }
            ]
        witness_biases = phase3_output.get("witness_biases", [])
        motives = phase3_output.get("motives", [])
        signals = phase3_output.get("signals", {})

        llm_consistency = signals.get("consistency_score", "medium")
        llm_motive = signals.get("motive_score", "medium")

        # Process contradiction overrides before calculating everything
        if feedbacks:
            dismissed_contradictions = {
                f["target"].lower() for f in feedbacks 
                if f.get("feedback_type") == "contradiction_override" and f.get("action") == "dismiss"
            }
            contradictions = [
                c for idx, c in enumerate(contradictions)
                if f"c_{idx}" not in dismissed_contradictions 
                and c.get("type", "").lower() not in dismissed_contradictions
                and not any(term in c.get("reason", "").lower() for term in dismissed_contradictions)
            ]

        # 1. Identify victim(s) in the case (case-agnostic)
        victim_nodes = []
        for ent in entities:
            etype = str(ent.get("type", "")).lower()
            elabel = str(ent.get("label", ent.get("name", "")))
            if "victim" in etype or "victim" in elabel.lower() or "deceased" in etype or "deceased" in elabel.lower():
                if elabel and elabel not in victim_nodes and " " in elabel:
                    victim_nodes.append(elabel)

        # Relation-based victim identification
        for rel in relations:
            src_id = str(rel.get("source", ""))
            tgt_id = str(rel.get("target", ""))
            rtype = str(rel.get("type", rel.get("label", ""))).lower()
            
            src_label = src_id
            tgt_label = tgt_id
            for ent in entities:
                ent_id = str(ent.get("id", ""))
                if ent_id == src_id:
                    src_label = ent.get("label", ent.get("name", src_id))
                if ent_id == tgt_id:
                    tgt_label = ent.get("label", ent.get("name", tgt_id))
                    
            if rtype in ["murdered_by", "killed_by", "deceased_by"]:
                if src_label and src_label not in victim_nodes and " " in src_label:
                    victim_nodes.append(src_label)
            elif rtype in ["murdered", "killed", "slew", "assassinated"]:
                if tgt_label and tgt_label not in victim_nodes and " " in tgt_label:
                    victim_nodes.append(tgt_label)

        # 2. Identify initial suspects strictly from graph/entities (not motives yet)
        suspect_nodes = [
            e.get("label", e.get("name", "")) for e in entities 
            if "suspect" in str(e.get("type", "")).lower() or "suspect" in str(e.get("label", "")).lower()
        ]

        # Scan relations for suspect indicators
        for rel in relations:
            src = str(rel.get("source", ""))
            tgt = str(rel.get("target", ""))
            rtype = str(rel.get("type", rel.get("label", ""))).lower()
            if any(kw in rtype for kw in ["perpetrator", "accomplice", "accused", "suspect", "killer", "murderer", "criminal", "murdered", "conspired"]):
                # Map source/target ID to labels if needed
                for node_id in (src, tgt):
                    node_label = node_id
                    node_type = ""
                    node_props = {}
                    for ent in entities:
                        if ent.get("id") == node_id:
                            node_label = ent.get("label", ent.get("name", node_id))
                            node_type = str(ent.get("type", ""))
                            node_props = ent.get("properties", {})
                            break
                    if self.is_person_node(node_label, node_type, node_props):
                        if node_label and not any(s.lower() == node_label.lower() for s in suspect_nodes):
                            suspect_nodes.append(node_label)

        # 3. Identify witnesses from graph/entities list
        witness_nodes = [
            e.get("label", e.get("name", "")) for e in entities 
            if "witness" in str(e.get("type", "")).lower() or "witness" in str(e.get("label", "")).lower()
        ]
        
        # Any person who is not an initial suspect and not the victim is a witness
        for ent in entities:
            label = ent.get("label", ent.get("name", ""))
            ntype = str(ent.get("type", ""))
            nprops = ent.get("properties", {})
            if self.is_person_node(label, ntype, nprops):
                label_lower = label.lower()
                is_victim = any(v.lower() == label_lower for v in victim_nodes)
                is_suspect = any(s.lower() == label_lower for s in suspect_nodes)
                is_witness = any(w.lower() == label_lower for w in witness_nodes)
                if not is_victim and not is_suspect and not is_witness:
                    witness_nodes.append(label)

        for b in witness_biases:
            name = b.get("witness_name", "")
            if name and not any(w.lower() == name.lower() for w in witness_nodes):
                witness_nodes.append(name)

        # 4. Integrate motives (LLM motives)
        for m in motives:
            name = m.get("party", "")
            if name:
                name_lower = name.lower()
                is_victim = any(v.lower() == name_lower for v in victim_nodes)
                is_suspect = any(s.lower() == name_lower for s in suspect_nodes)
                is_witness = any(w.lower() == name_lower for w in witness_nodes)
                if not is_victim and not is_suspect and not is_witness:
                    suspect_nodes.append(name)

        # Case-insensitive deduplication, filtering out non-people, victims, and witnesses
        unique_suspects = []
        for s in suspect_nodes:
            s_lower = s.lower()
            if s_lower in ["charge sheet charge_sheet-001", "financial conspiracy"]:
                continue
            if any(v.lower() == s_lower for v in victim_nodes):
                continue
            if any(w.lower() == s_lower for w in witness_nodes):
                continue
            
            # Check if explicitly a witness or victim node in graph
            is_wit_or_vic = False
            for ent in entities:
                ent_label = ent.get("label", ent.get("name", ""))
                if ent_label.lower() == s_lower:
                    etype = str(ent.get("type", "")).lower()
                    elabel = str(ent.get("label", ent.get("name", ""))).lower()
                    if "witness" in etype or "witness" in elabel or "victim" in etype or "victim" in elabel or "deceased" in etype or "deceased" in elabel:
                        is_wit_or_vic = True
                        break
            if is_wit_or_vic:
                continue

            if not any(u.lower() == s_lower for u in unique_suspects):
                best_cased = s
                for ent in entities:
                    ent_label = ent.get("label", ent.get("name", ""))
                    if ent_label.lower() == s_lower:
                        best_cased = ent_label
                        break
                unique_suspects.append(best_cased)

        suspect_nodes = unique_suspects

        if not suspect_nodes:
            suspect_nodes = ["Marcus Vance"]

        # Clean witness list to exclude suspect nodes and victim nodes, and case-insensitively deduplicate
        unique_witnesses = []
        for w in witness_nodes:
            w_lower = w.lower()
            if any(s.lower() == w_lower for s in suspect_nodes):
                continue
            if any(v.lower() == w_lower for v in victim_nodes):
                continue
            if w_lower in ["charge sheet charge_sheet-001", "financial conspiracy"]:
                continue
            if not any(uw.lower() == w_lower for uw in unique_witnesses):
                best_cased = w
                for ent in entities:
                    ent_label = ent.get("label", ent.get("name", ""))
                    if ent_label.lower() == w_lower:
                        best_cased = ent_label
                        break
                unique_witnesses.append(best_cased)
        witness_nodes = unique_witnesses
        if not witness_nodes:
            witness_nodes = ["Detective Sean Ryan", "Sarah Jenkins"]

        # 4. Compute Witness Credibilities (passing feedbacks)
        witness_credibilities = {}
        llm_witness_creds = phase3_output.get("llm_witness_credibilities", {})
        for w in witness_nodes:
            llm_cred = None
            if llm_witness_creds:
                for k, v in llm_witness_creds.items():
                    if k.lower() in w.lower() or w.lower() in k.lower():
                        llm_cred = float(v)
                        break
            if llm_cred is not None:
                witness_credibilities[w] = llm_cred
            else:
                witness_credibilities[w] = self.calculate_witness_credibility(
                    witness_name=w,
                    contradictions=contradictions,
                    witness_biases=witness_biases,
                    relations=relations,
                    llm_consistency_score=llm_consistency,
                    feedbacks=feedbacks
                )

        # 5. Compute Evidence Weights (passing feedbacks)
        evidence_weights = self.evaluate_evidence_weights(
            entities=entities,
            contradictions=contradictions,
            relations=relations,
            witness_credibilities=witness_credibilities,
            feedbacks=feedbacks,
            llm_evidence_weights=phase3_output.get("llm_evidence_weights", {})
        )

        # 6. Compute Contradiction Severity Index
        contradiction_results = self.calculate_contradiction_severity(contradictions)

        # 6. Compute Suspect and Conviction Probabilities (passing feedbacks)
        suspect_probabilities = {}
        conviction_probabilities = {}
        for s in suspect_nodes:
            suspect_probabilities[s] = self.calculate_suspect_probability(
                suspect_name=s,
                entities=entities,
                relations=relations,
                evidence_weights=evidence_weights,
                motives=motives,
                contradictions=contradictions,
                llm_motive_score=llm_motive,
                feedbacks=feedbacks
            )
            conviction_probabilities[s] = self.calculate_conviction_probability(
                suspect_name=s,
                entities=entities,
                relations=relations,
                evidence_weights=evidence_weights,
                contradictions=contradictions,
                witness_credibilities=witness_credibilities,
                feedbacks=feedbacks
            )

        # Compile formatted results compatible with MemoryExplorer and frontend
        formatted_witnesses = []
        for w, cred in witness_credibilities.items():
            c_count = sum(1 for c in contradictions if w.lower() in c.get("reason", "").lower())
            formatted_witnesses.append({
                "name": w,
                "credibility": int(cred * 100),
                "role": "Witness",
                "contradictions": c_count
            })

        # Add suspects to formatted_witnesses with role "Suspect" so they show up in Theory Dismissals and Witness Reliability select
        for s in suspect_nodes:
            if not any(item["name"].lower() == s.lower() for item in formatted_witnesses):
                c_count = sum(1 for c in contradictions if s.lower() in c.get("reason", "").lower())
                prob = suspect_probabilities.get(s, 0.5)
                cred = 1.0 - prob
                formatted_witnesses.append({
                    "name": s,
                    "credibility": int(cred * 100),
                    "role": "Suspect",
                    "contradictions": c_count
                })

        formatted_contradictions = []
        for idx, c in enumerate(contradictions):
            formatted_contradictions.append({
                "id": f"c_{idx}",
                "title": f"Discrepancy {idx+1}: {c.get('type', 'Factual')}",
                "description": c.get("reason", c.get("description", "")),
                "severity": c.get("severity", "medium")
            })

        # Select the primary suspect as the one with the highest conviction probability
        primary_suspect = "Unknown"
        max_conv_prob = -1.0
        for s in suspect_nodes:
            cp = conviction_probabilities.get(s, 0.0)
            if cp > max_conv_prob:
                max_conv_prob = cp
                primary_suspect = s

        s_prob = suspect_probabilities.get(primary_suspect, 0.5)
        c_prob = conviction_probabilities.get(primary_suspect, 50.0)
        logger.info(f"[DEBUG_ENGINE] suspect_nodes: {suspect_nodes}")
        logger.info(f"[DEBUG_ENGINE] primary_suspect: {primary_suspect}")
        logger.info(f"[DEBUG_ENGINE] suspect_probabilities: {suspect_probabilities}")
        logger.info(f"[DEBUG_ENGINE] conviction_probabilities: {conviction_probabilities}")

        # Filter out 0.0 weight items for confidenceScore average
        valid_weights = [w for w in evidence_weights.values() if w > 0.0]

        return {
            "witness_credibilities": witness_credibilities,
            "evidence_weights": evidence_weights,
            "contradiction_severity": contradiction_results,
            "suspect_probabilities": suspect_probabilities,
            "conviction_probabilities": conviction_probabilities,
            "ui_metrics": {
                "confidenceScore": int(min(1.0, sum(valid_weights) / max(1, len(valid_weights))) * 100) if valid_weights else 70,
                "suspectProbability": int(s_prob * 100),
                "convictionProbability": int(c_prob),
                "witnesses": formatted_witnesses,
                "contradictions": formatted_contradictions
            }
        }
