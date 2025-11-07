from .data_preprocessing import valid_categories



def show_valid_options():
    print("\n Valid options for each feature:")
    print("=" * 50)
    for feature, options in valid_categories.items():
        print(f"{feature}: {', '.join(options)}")
    print("=" * 50)

def validate_input(feature, value):
    """Check if input value is valid for the given feature"""
    if feature in valid_categories:
        return value in valid_categories[feature]
    return True

def get_closest_match(feature, value):
    """Find closest match for invalid input"""
    if feature not in valid_categories:
        return value
    valid_options = valid_categories[feature]
    value_lower = value.lower()
    for option in valid_options:
        if option.lower() == value_lower:
            return option
    for option in valid_options:
        if value_lower in option.lower() or option.lower() in value_lower:
            return option
    return None
