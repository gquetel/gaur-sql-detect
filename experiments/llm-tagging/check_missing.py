"""
Verify which tags are missing.
"""
import csv

def read_rules_list():
    """Simple file with one rule per line:

    start_entry
    sql_statement
    835_1
    opt_end_of_input
    simple_statement_or_begin
    simple_statement
    deallocate
    deallocate_or_drop
    prepare
    [...]

    """
    path = "./data/rules.txt"
    with open(path, "r", encoding="utf-8") as f:
        rules = [line.strip() for line in f if line.strip()]
    return rules


def read_tags_list():
    """ 
    CSV file with 2 columns: rulename and tag. Obtained from tagging using LLMs:

    rulename,tag
    start_entry,STATEMENT_CONTROL
    sql_statement,STATEMENT_CONTROL
    835_1,STATEMENT_CONTROL
    opt_end_of_input,STATEMENT_CONTROL
    simple_statement_or_begin,STATEMENT_CONTROL
    simple_statement,STATEMENT_CONTROL
    deallocate,STATEMENT_MANAGEMENT
    deallocate_or_drop,STATEMENT_MANAGEMENT
    prepare,STATEMENT_MANAGEMENT
    prepare_src,STATEMENT_MANAGEMENT
    [...]
    """
    path = "./data/claude-tags.csv"
    tagged_rules = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tagged_rules.add(row["rulename"].strip())
    return tagged_rules


def find_missing(rules, tagged_rules):
    """Find all missing rules in tagged_rules

    Args:
        rules (_type_): _description_
        tagged_rules (_type_): _description_
    """
    missing = [rule for rule in rules if rule not in tagged_rules]
    return missing

if __name__ == "__main__":
    rules = read_rules_list()
    tagged_rules = read_tags_list()
    missing_rules = find_missing(rules, tagged_rules)

    print("Rules without tags:")
    for rule in missing_rules:
        print(rule)
