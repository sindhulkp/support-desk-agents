"""
agents/order_lookup.py — a TOOL: look up a real order by its number.

This is what turns the desk from a demo of a pretend shop into a tool
that can answer "where is MY order?" about an actual order. The shop's
orders live in data/orders.json — order numbers, items, totals, and the
live status (Processing / Shipped / Out for delivery / Delivered /
Delayed), with carrier and tracking number.

Two jobs, both small:
  1. extract_order_id(text) — find "ORD-1001" or "#1001" in a ticket
     and normalise it to "ORD-1001". A bare number ("I have 2 orders")
     is NOT an order id — matching that would look up the wrong order.
  2. find_order(order_id) — return the order dict, or None when no
     order with that number exists. None is a real answer: the
     responder must then say the order could not be found, never
     invent a status.

format_order(order) renders one order as a short block of facts for
the responder's prompt — status, items, total, carrier, tracking
number, estimated delivery. The responder is only allowed to state
facts from this block (or the retrieved articles); it never guesses
a delivery date.
"""

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDERS_FILE = os.path.join(BASE_DIR, "data", "orders.json")

# "ORD-1001" or "#1001" (case-insensitive) — the two ways customers
# write an order number. A bare "1001" is not matched on purpose:
# "I ordered 2 items" must never become an order lookup.
ORDER_ID_RE = re.compile(r"(?:ORD-?|#)(\d+)", re.IGNORECASE)


def extract_order_id(text):
    """Return the first order id in the text as "ORD-1001", or None."""
    match = ORDER_ID_RE.search(text or "")
    if not match:
        return None
    return f"ORD-{match.group(1)}"


def load_orders(path=None):
    """Load the orders database (a list of order dicts)."""
    path = path or ORDERS_FILE
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_order(order_id, orders=None):
    """Look up one order by id. Returns the order dict, or None when
    no order with that number exists. None is not an error — it is
    the honest answer the responder has to give."""
    if not order_id:
        return None
    orders = orders if orders is not None else load_orders()
    for order in orders:
        if order["order_id"].upper() == order_id.upper():
            return order
    return None


def format_order(order):
    """Render one order as a short block of facts for the responder.

    One line per fact, in a fixed shape — the mock LLM parses these
    lines back out, so keep the labels exactly as they are.
    """
    items = ", ".join(
        f"{item['name']} x{item['qty']} (${item['price']:.2f})"
        for item in order["items"]
    )
    return (
        f"Order {order['order_id']} — status: {order['status']}\n"
        f"Customer: {order['customer_name']}\n"
        f"Items: {items}\n"
        f"Total: ${order['total']:.2f}\n"
        f"Carrier: {order['carrier']}, tracking {order['tracking_number']}\n"
        f"Estimated delivery: {order['estimated_delivery']}"
    )
