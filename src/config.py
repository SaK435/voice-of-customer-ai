MODEL_NAME = "qwen2.5:3b"

ASPECT_DEFINITIONS = {
    "Effectiveness": "whether the product does its job: cleans, moisturises, styles, removes hair, colour lasts, nails cure",
    "Appearance and colour": "how the product looks: colour, shade, design, how it looks once applied",
    "Quality and build": "how well made the product feels: sturdy, flimsy, cheap, well made",
    "Durability": "how long the item lasts before breaking, wearing out or stopping working",
    "Scent": "the smell of the product",
    "Texture and feel": "how the product feels physically: consistency, softness, stiffness, greasiness",
    "Skin or hair reaction": "effects on the body: irritation, rash, breakouts, dryness, allergic reaction",
    "Ingredients and ethics": "natural, organic, cruelty free, vegan, chemicals, ingredient list",
    "Packaging": "bottle, box, pump, seal, leaks, broken container, labelling, excessive packaging",
    "Price and value": "cost, worth the money, cheaper or more expensive than alternatives",
    "Size and quantity": "amount of product, size of the item, how many pieces in the pack",
    "Ease of use": "how easy it is to apply, put on, remove, wash out or clean. NOT whether it works",
    "Shipping and delivery": "arrival time, damaged in transit, wrong item sent",
    "Customer service": "seller responses, returns, refunds, questions answered or ignored",
    "Authenticity": "ONLY when the reviewer says the product is fake, counterfeit or not the genuine brand",
    "Other": "a specific product detail that fits none of the categories above",
}

ASPECTS = list(ASPECT_DEFINITIONS)