import json
import re
import os
# --------------------------------------------------------------------
# FULL NORMALIZATION MAP (one canonical relation per cluster)
# --------------------------------------------------------------------

NORMALIZATION_MAP = {
    # -----------------------------
    # Life Events & Personal Status
    # -----------------------------
    r"^(born|born_as|born_on|born_in)$": "born_in",
    r"^(died_on|died_in|killed_at_age)$": "died_in",
    r"^(married|married_to|married_in|married_on|married_at)$": "married_to",
    r"^(divorced|divorced_on|remarried)$": "divorced",
    r"^(pregnant|pregnant_during)$": "pregnant",
    r"^(lived_in|lived_with|lived_near|lived_until)$": "lived_in",
    r"^(lives_with|lives_as)$": "lives_with",
    r"^(age_of)$": "age_of",
    r"^(personal_status)$": "personal_status",
    r"^(personal_attribute)$": "personal_attribute",
    r"^(characteristic_of)$": "characteristic_of",
    r"^(name_change|changed_name)$": "name_change",
    r"^(was|were|became)$": "was",
    r"^(called|was_called|named|named_after)$": "called",

    # -----------------------------
    # Family & Personal Relations
    # -----------------------------
    r"^(family_of)$": "family_of",
    r"^(family_relation|family_relationship)$": "family_relation",
    r"^(parent_of)$": "parent_of",
    r"^(friend_from)$": "friend_from",
    r"^(met|met_at|met_in|met_on|met_through|met_with)$": "met",
    r"^(knew)$": "knew",
    r"^(encountered)$": "encountered",
    r"^(helped|helped_by)$": "helped",
    r"^(supported|supported_by|supported_each_other)$": "supported",
    r"^(rescued_from)$": "rescued_from",
    r"^(reunited_with|reunited_at)$": "reunited_with",
    r"^(concerned_about)$": "concerned_about",
    r"^(afraid_of)$": "afraid_of",
    r"^(felt)$": "felt",
    r"^(remember)$": "remember",

    # -----------------------------
    # Health, Injury, Medical
    # -----------------------------
    r"^(illness|ill_in|illness_on)$": "illness",
    r"^(became_ill|fell_ill_in)$": "became_ill",
    r"^(health_event)$": "health_event",
    r"^(hospitalized_in|hospitalized_for|hospitalized_with)$": "hospitalized_in",
    r"^(injured|injured_in|injured_by)$": "injured",
    r"^(wounded_in|wounded_on|wounded_near)$": "wounded_in",
    r"^(unhealthy_in)$": "unhealthy_in",
    r"^(recovered_from)$": "recovered_from",
    r"^(suffered_from|suffered_in)$": "suffered_from",

    # -----------------------------
    # Education & Professional Life
    # -----------------------------
    r"^(education)$": "education",
    r"^(achieved_degree)$": "achieved_degree",
    r"^(studied_at|studied_in|studied_until|studied_as|studied_with)$": "studied_at",
    r"^(graduated_from|graduated_in|graduated_on)$": "graduated_from",
    r"^(worked|worked_as|worked_at|worked_for|worked_in|worked_on|worked_from|worked_until|worked_since|worked_with)$": "worked_in",
    r"^(member_of)$": "member_of",
    r"^(joined|joined_in|joined_org|joined_organization|joined_entity|joined_army_on)$": "joined",
    r"^(served_in|served_as|served_for|served_until|served_with)$": "served_in",
    r"^(trained_with)$": "trained_with",
    r"^(researched_at)$": "researched_at",
    r"^(passed_exam)$": "passed_exam",
    r"^(authored|published)$": "authored",

    # -----------------------------
    # Movement & Personal Actions
    # -----------------------------
    r"^(came|came_from|came_to_rescue)$": "came",
    r"^(went_to)$": "went_to",
    r"^(arrived_at|arrived_on|arrived_from|arrived_in)$": "arrived_at",
    r"^(left|left_in|left_on|left_from)$": "left",
    r"^(moved_from|moved_to)$": "moved_to",
    r"^(planned_to|planned_to_move)$": "planned_to",
    r"^(intended_to_move)$": "intended_to_move",
    r"^(traveled|traveled_to|traveled_with|traveled_on|traveled_over|traveled_through)$": "traveled",
    r"^(commuted_to)$": "commuted_to",
    r"^(ran_to)$": "ran_to",
    r"^(reached)$": "reached",
    r"^(returned|returned_on|returned_from|returned_to|returned_after|returned_home)$": "returned",
    r"^(sent_home)$": "sent_home",

    # -----------------------------
    # Legal, Social, Emotional
    # -----------------------------
    r"^(admitted_to)$": "admitted_to",
    r"^(accused_of)$": "accused_of",
    r"^(investigated)$": "investigated",
    r"^(questioned_about)$": "questioned_about",
    r"^(refused)$": "refused",
    r"^(requested|requested_return)$": "requested",
    r"^(reported_to|reported_by)$": "reported_to",
    r"^(warned_about)$": "warned_about",
    r"^(selected_by|selected_for)$": "selected_by",
    r"^(suspected_as)$": "suspected_as",
    r"^(victim_of)$": "victim_of",
    r"^(punished)$": "punished",
    r"^(insulted_for)$": "insulted_for",
    r"^(harassed_by)$": "harassed_by",
    r"^(depressed_by)$": "depressed_by",
    r"^(shocked_by)$": "shocked_by",

    # -----------------------------
    # Geographical Location & Presence
    # -----------------------------
    r"^(located_at|located_in)$": "located_in",
    r"^(from|from_location)$": "from_location",
    r"^(live_in|lived_in|lived_near)$": "lived_in",
    r"^(born_in)$": "born_in",
    r"^(worked_in)$": "worked_in",
    r"^(studied_in)$": "studied_in",
    r"^(buried_in)$": "buried_in",
    r"^(hidden_in|hid_in)$": "hidden_in",
    r"^(kept_in)$": "kept_in",
    r"^(stayed_in|stayed_with)$": "stayed_in",
    r"^(met_at|met_in)$": "met_at",
    r"^(socialized_in)$": "socialized_in",
    r"^(occupied_in)$": "occupied_in",
    r"^(overcrowded_in)$": "overcrowded_in",
    r"^(established_in)$": "established_in",
    r"^(infested_in)$": "infested_in",
    r"^(starved_in)$": "starved_in",
    r"^(struggled_in)$": "struggled_in",
    r"^(tolerant_in)$": "tolerant_in",
    r"^(dangerous_in)$": "dangerous_in",

    # -----------------------------
    # Migration & Forced Movement
    # -----------------------------
    r"^(emigrated_to|emigrated_on)$": "emigrated_to",
    r"^(immigrated_to)$": "immigrated_to",
    r"^(attempted_emigration)$": "attempted_emigration",
    r"^(attempted_escape_to)$": "attempted_escape_to",
    r"^(escaped_from|escaped_to|escaped_via|escaped_with|escaped_on)$": "escaped_from",
    r"^(evacuated_from|evacuated_to|evacuated_on|evacuated_with)$": "evacuated_from",
    r"^(resettled_to)$": "resettled_to",
    r"^(relocated_to)$": "relocated_to",
    r"^(sent_from|sent_to)$": "sent_from",
    r"^(transported_from|transported_to|transported_through|transported_by)$": "transported_from",

    # -----------------------------
    # Events Linked to Places
    # -----------------------------
    r"^(occurred_in|occurred_on|occurred_at)$": "occurred_in",
    r"^(battle_at)$": "battle_at",
    r"^(bombed_in|bombed_on)$": "bombed_in",
    r"^(killed_in|killed_near|killed_during)$": "killed_in",
    r"^(arrested_in|arrested_on)$": "arrested_in",
    r"^(captured_in)$": "captured_in",
    r"^(deported_from|deported_on|deported_to)$": "deported_from",
    r"^(interned_to)$": "interned_to",
    r"^(stationed_in)$": "stationed_in",
    r"^(sank_in)$": "sank_in",
    r"^(searched_in)$": "searched_in",
    r"^(tattooed_in|tattooed_on)$": "tattooed_in",
    r"^(worked_at|worked_for)$": "worked_at",

    # -----------------------------
    # Conflict, War, Violence
    # -----------------------------
    r"^(war_began|war_began_on)$": "war_began",
    r"^(war_started|war_started_on)$": "war_started",
    r"^(armed|armed_with)$": "armed",
    r"^(fought_against|fought_in|fought_at)$": "fought_against",
    r"^(attacked|attacked_by|attacked_at|attacked_in)$": "attacked",
    r"^(attack_on)$": "attack_on",
    r"^(bombed|bombed_by)$": "bombed",
    r"^(captured_by)$": "captured_by",
    r"^(killed_by)$": "killed_by",
    r"^(surrendered_to)$": "surrendered_to",
    r"^(liberated_by|liberated_from|liberated_on)$": "liberated_by",
    r"^(protected_by|protected_against|protected_during)$": "protected_by",
    r"^(under_attack)$": "under_attack",

    # -----------------------------
    # Persecution & Oppression
    # -----------------------------
    r"^(persecuted_as|persecuted_by|persecuted_in)$": "persecuted_by",
    r"^(discriminated_against|discriminated_in|discriminated_at|discriminated_for)$": "discriminated_against",
    r"^(forced_labor_in)$": "forced_labor_in",
    r"^(forced_to_move)$": "forced_to_move",
    r"^(forced_to_wear)$": "forced_to_wear",
    r"^(evicted_by)$": "evicted_by",
    r"^(expelled_from)$": "expelled_from",
    r"^(confiscated|confiscated_from|confiscated_in)$": "confiscated",
    r"^(censored|censored_during)$": "censored",
    r"^(unjustly_detained|unjustly_imprisoned)$": "unjustly_imprisoned",
    r"^(occupied|occupied_by|occupied_on)$": "occupied",
    r"^(restricted_by)$": "restricted_by",
    r"^(limited_by)$": "limited_by",
    r"^(purge)$": "purge",

    # -----------------------------
    # Arrests, Imprisonment, Deportation
    # -----------------------------
    r"^(arrested|arrested_again|arrested_by|arrested_for)$": "arrested",
    r"^(imprisoned_in|imprisoned_from|imprisoned_until|imprisoned_with|imprisoned_for)$": "imprisoned_in",
    r"^(deported|deported_from|deported_on|deported_to|deported_with)$": "deported",
    r"^(interned_with|interned_to)$": "interned_with",
    r"^(kept_in)$": "kept_in",
    r"^(under_attack)$": "under_attack",
    r"^(unjustly_imprisoned)$": "unjustly_imprisoned",

    # -----------------------------
    # Historical Events, Politics
    # -----------------------------
    r"^(annexed_by)$": "annexed_by",
    r"^(governed_by)$": "governed_by",
    r"^(controlled_by)$": "controlled_by",
    r"^(implemented_by)$": "implemented_by",
    r"^(organized_by|was_organized)$": "organized_by",
    r"^(established_on)$": "established_on",
    r"^(liquidated_on)$": "liquidated_on",
    r"^(revived_in)$": "revived_in",
    r"^(operated_until)$": "operated_until",

    # -----------------------------
    # Historical Social Patterns
    # -----------------------------
    r"^(common_practice)$": "common_practice",
    r"^(dangerous_activity)$": "dangerous_activity",
    r"^(getting_organized)$": "getting_organized",
    r"^(personal_status)$": "personal_status",
}


# --------------------------------------------------------------------
# NORMALIZATION FUNCTION
# --------------------------------------------------------------------
def normalize_relation(rel):
    for pattern, canonical in NORMALIZATION_MAP.items():
        if re.match(pattern, rel):
            return canonical
    return rel  # unchanged if no match

# --------------------------------------------------------------------
# PROCESS ALL JSON FILES IN data1/
# --------------------------------------------------------------------
def normalize_folder(input_folder="processed_data", output_folder="normalized"):

    # Create output folder if missing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    print(f"Found {len(files)} JSON files in '{input_folder}/'")

    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Normalize relationships
        for rel in data.get("extracted_relationships", []):
            rel["relationship_type"] = normalize_relation(rel["relationship_type"])

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"→ Normalized: {file}")

    print("\nDone! Normalized files saved in 'normalized/' folder.\n")


# Run automatically
if __name__ == "__main__":
    normalize_folder()