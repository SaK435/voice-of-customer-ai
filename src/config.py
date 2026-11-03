MODEL_NAME = "qwen2.5:3b"

ASPECT_DEFINITIONS = {
    "Effectiveness": "whether the product does its job (cleans, moisturises, styles, removes hair)",
    "Scent": "the smell of the product",
    "Texture and feel": "how the product feels physically: consistency, softness, stiffness, greasiness",
    "Skin or hair reaction": "effects on the body: irritation, rash, breakouts, dryness, allergic reaction",
    "Packaging": "bottle, box, pump, seal, leaks, broken container",
    "Price and value": "cost, worth the money, cheaper or more expensive than alternatives",
    "Size and quantity": "amount of product, size of the item, how long it lasts before running out",
    "Ease of use": "how easy it is to apply, use, clean or install",
    "Durability": "how long the item itself lasts before breaking or wearing out",
    "Shipping and delivery": "arrival time, damaged in transit, wrong item sent",
    "Authenticity": "fake, counterfeit, different from the original version",
    "Other": "anything that does not fit the categories above",
}

ASPECTS = list(ASPECT_DEFINITIONS)