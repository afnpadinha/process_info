from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from .rng import RNG
from .spin_engine import SpinEngine
from . import config

# Path to the frontend directory (use absolute path to avoid access issues)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'front_end'

app = Flask(__name__)
CORS(app)  # permite pedidos do front-end (http://127.0.0.1:5500, etc.)

print(f"[DEBUG] Frontend directory: {FRONTEND_DIR}")
print(f"[DEBUG] Frontend exists: {FRONTEND_DIR.exists()}")

# cria um motor de roleta único para o servidor
rng = RNG()
spin_engine = SpinEngine(rng, config)


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"message": "backend OK"})


def evaluate_bet(bet_type: str, landing_number: int) -> tuple:
    """
    Evaluate a bet and return (is_winner, payout_multiplier).
    
    Bet types from frontend:
    - "0" to "36": single number bets (35:1)
    - "red", "black": color bets (1:1)
    - "par", "impar": even/odd bets (1:1)
    - "1-18", "19-36": low/high bets (1:1)
    - "1st12", "2nd12", "3rd12": dozen bets (2:1)
    - "col1", "col2", "col3": column bets (2:1)
    """
    # Single number bet (0-36)
    if bet_type.isdigit() or bet_type == "0":
        number = int(bet_type)
        if number == landing_number:
            return True, 35  # 35:1 payout (plus original bet returned)
        return False, 0
    
    # Color bets
    if bet_type == "red":
        red_numbers = config.COLOR_MAP["1"]
        return landing_number in red_numbers, 1
    if bet_type == "black":
        black_numbers = config.COLOR_MAP["2"]
        return landing_number in black_numbers, 1
    
    # Even/Odd bets (0 loses)
    if bet_type == "par":  # even
        if landing_number == 0:
            return False, 0
        return landing_number % 2 == 0, 1
    if bet_type == "impar":  # odd
        if landing_number == 0:
            return False, 0
        return landing_number % 2 == 1, 1
    
    # Low/High bets (0 loses)
    if bet_type == "1-18":
        return 1 <= landing_number <= 18, 1
    if bet_type == "19-36":
        return 19 <= landing_number <= 36, 1
    
    # Dozen bets (0 loses)
    if bet_type == "1st12":
        return landing_number in config.DOZEN_MAP["1"], 2
    if bet_type == "2nd12":
        return landing_number in config.DOZEN_MAP["2"], 2
    if bet_type == "3rd12":
        return landing_number in config.DOZEN_MAP["3"], 2
    
    # Column bets (0 loses)
    if bet_type == "col1":
        return landing_number in config.COLUMNS["1"], 2
    if bet_type == "col2":
        return landing_number in config.COLUMNS["2"], 2
    if bet_type == "col3":
        return landing_number in config.COLUMNS["3"], 2
    
    # Unknown bet type
    return False, 0


@app.route("/api/spin", methods=["POST"])
def spin():
    """
    Expects JSON with an array of bets:
    {
      "bets": [
        {"betType": "17", "amount": 10, "number": 17},
        {"betType": "red", "amount": 5, "number": null},
        {"betType": "1st12", "amount": 10, "number": null}
      ]
    }
    
    Also supports legacy single-bet format for backwards compatibility:
    {
      "amount": 10,
      "number": 17
    }
    """
    data = request.get_json(force=True) or {}
    
    # 1) Spin the wheel
    slot_index = spin_engine.simulate()           # index 0..36
    landing_number = config.WHEEL[slot_index]     # actual roulette number
    
    # 2) Handle bets - support both array format and legacy single-bet format
    bets_input = data.get("bets", [])
    
    # Legacy format support
    if not bets_input and "number" in data:
        bets_input = [{
            "betType": str(data.get("number", -1)),
            "amount": float(data.get("amount", 0)),
            "number": data.get("number")
        }]
    
    # 3) Evaluate each bet
    total_bet = 0.0
    total_payout = 0.0
    results = []
    any_win = False
    
    for bet in bets_input:
        bet_type = bet.get("betType", "")
        amount = float(bet.get("amount", 0))
        total_bet += amount
        
        is_winner, multiplier = evaluate_bet(bet_type, landing_number)
        
        if is_winner:
            # Payout = bet amount * multiplier + original bet returned
            payout = amount * multiplier + amount
            any_win = True
        else:
            payout = 0.0
        
        total_payout += payout
        
        results.append({
            "betType": bet_type,
            "amount": amount,
            "win": is_winner,
            "payout": payout
        })
    
    return jsonify({
        "landing_slot_index": slot_index,
        "landing_number": landing_number,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "win": any_win,
        "results": results
    })


# Static file serving routes (defined after API routes to avoid conflicts)
@app.route("/")
def serve_index():
    """Serve the frontend index.html"""
    index_path = FRONTEND_DIR / 'index.html'
    print(f"[DEBUG] Serving index from: {index_path}")
    return send_file(index_path)


@app.route("/<path:path>")
def serve_static(path):
    """Serve static files from the frontend directory"""
    # Skip API routes (they're handled above)
    if path.startswith('api/'):
        return jsonify({"error": "Not found"}), 404
    file_path = FRONTEND_DIR / path
    print(f"[DEBUG] Serving static file: {file_path}")
    if file_path.exists() and file_path.is_file():
        return send_file(file_path)
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    # correr como módulo: python -m backend.server
    # Using port 5001 to avoid conflict with macOS AirPlay Receiver on port 5000
    app.run(host='0.0.0.0', port=5001, debug=True)
