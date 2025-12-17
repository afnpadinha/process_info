// script.js

const RED_NUMBERS = new Set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]);
const ROULETTE_ORDER = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26];

const BALL_RADIUS = 135;

// Colunas da roleta (para col1, col2, col3)
const COLUMN_MAP = {
    col1: new Set([1,4,7,10,13,16,19,22,25,28,31,34]),
    col2: new Set([2,5,8,11,14,17,20,23,26,29,32,35]),
    col3: new Set([3,6,9,12,15,18,21,24,27,30,33,36])
};

// Calcula o payout de UMA aposta, dado o número vencedor
function payoutForBet(bet, winningNumber) {
    const a = bet.amount;

    // Se saiu 0, só ganha quem apostou no 0
    if (winningNumber === 0) {
        if (bet.number === 0) {
            return a * 36; // número direito paga 36x
        }
        return 0;
    }

    // 1) Aposta em número direto
    if (bet.number !== null && bet.number !== undefined) {
        return (winningNumber === bet.number) ? a * 36 : 0;
    }

    const t = bet.betType; // ex: "red", "par", "1-18", "1st12", "col1"

    // 2) Cores
    if (t === "red") {
        return RED_NUMBERS.has(winningNumber) ? a * 2 : 0;
    }
    if (t === "black") {
        return (!RED_NUMBERS.has(winningNumber) && winningNumber !== 0) ? a * 2 : 0;
    }

    // 3) Par / Ímpar
    if (t === "par") {
        return (winningNumber % 2 === 0) ? a * 2 : 0;
    }
    if (t === "impar") {
        return (winningNumber % 2 === 1) ? a * 2 : 0;
    }

    // 4) 1-18 / 19-36
    if (t === "1-18") {
        return (winningNumber >= 1 && winningNumber <= 18) ? a * 2 : 0;
    }
    if (t === "19-36") {
        return (winningNumber >= 19 && winningNumber <= 36) ? a * 2 : 0;
    }

    // 5) Dezenas
    if (t === "1st12") {
        return (winningNumber >= 1 && winningNumber <= 12) ? a * 3 : 0;
    }
    if (t === "2nd12") {
        return (winningNumber >= 13 && winningNumber <= 24) ? a * 3 : 0;
    }
    if (t === "3rd12") {
        return (winningNumber >= 25 && winningNumber <= 36) ? a * 3 : 0;
    }

    // 6) Colunas
    if (t === "col1") {
        return COLUMN_MAP.col1.has(winningNumber) ? a * 3 : 0;
    }
    if (t === "col2") {
        return COLUMN_MAP.col2.has(winningNumber) ? a * 3 : 0;
    }
    if (t === "col3") {
        return COLUMN_MAP.col3.has(winningNumber) ? a * 3 : 0;
    }

    // Se for um tipo que ainda não tratámos
    return 0;
}


let currentBallAngle = 0;
let ballElement = null;

// 🔹 Estado das apostas
let bets = [];          // apostas atuais na mesa
let lastRoundBets = []; // para REAPOSTAR

// Define a posição da bola em função de um ângulo (em graus)
function setBallAngle(angleDeg) {
    if (!ballElement) return;
    ballElement.style.transform =
        `translate(-50%, -50%) rotate(${angleDeg}deg) translateY(-${BALL_RADIUS}px)`;
}

// Anima a bola até ao slot indicado (index na ROULETTE_ORDER)
function spinBallToSlot(slotIndex, durationSeconds) {
    if (!ballElement) return;

    const segmentAngle = 360 / ROULETTE_ORDER.length;
    const targetAngle = slotIndex * segmentAngle + segmentAngle / 2;

    // Faz algumas voltas completas antes de parar
    const extraSpins = 4;
    const normalizedCurrent = currentBallAngle % 360;
    const deltaToTarget = ((targetAngle - normalizedCurrent) + 360) % 360;
    const finalAngle = currentBallAngle + extraSpins * 360 + deltaToTarget;

    ballElement.style.transition =
        `transform ${durationSeconds}s cubic-bezier(0.1, 0.7, 1.0, 0.1)`;
    setBallAngle(finalAngle);

    currentBallAngle = finalAngle;
}

document.addEventListener('DOMContentLoaded', () => {
    const chipSelector = document.querySelector('.chip-selector');
    const totalBetDisplay = document.getElementById('total-bet-display');
    const numberCellsContainer = document.querySelector('.number-cells');
    const wheelInnerRing = document.querySelector('.wheel-inner-ring');
    const wheelNumberMarkers = document.querySelector('.wheel-number-markers');
    ballElement = document.querySelector('.ball');
    setBallAngle(0); // posição inicial da bola

    const balanceDisplay = document.getElementById('balance-display');
    const lastWinDisplay = document.getElementById('last-win-display');

    // construir mesa e roda
    populateNumberCells(numberCellsContainer);
    setupWheel(wheelInnerRing, wheelNumberMarkers);

    const bettingCells = document.querySelectorAll('#betting-grid .cell');

    let selectedChipValue = 1;
    let totalBet = 0.00;
    let currentNumberBet = null;

    // mostrar aposta total inicial a 0
    totalBetDisplay.textContent = totalBet.toFixed(2);

    // botões de ação
    const clearButton = document.getElementById('btn-clear');
    const undoButton = document.getElementById('btn-undo');
    const rebetButton = document.getElementById('btn-rebet');
    const doubleButton = document.getElementById('btn-double');

    // 🔸 Recalcular aposta total
    function recalcTotalBet() {
        totalBet = bets.reduce((sum, bet) => sum + bet.amount, 0);
        totalBetDisplay.textContent = totalBet.toFixed(2);
    }

    // 🔸 Limpar visualmente todas as fichas
    function clearBetsUI() {
        for (const bet of bets) {
            if (bet.chipElement && bet.chipElement.parentNode) {
                bet.chipElement.parentNode.removeChild(bet.chipElement);
            }
        }
        bets = [];
        currentNumberBet = null;
        recalcTotalBet();
    }

    // 🔸 Atualizar currentNumberBet com base na última aposta numérica
    function updateCurrentNumberBetFromBets() {
        const lastNumeric = [...bets].reverse().find(b => b.number !== null);
        if (lastNumeric) {
            currentNumberBet = {
                amount: lastNumeric.amount,
                number: lastNumeric.number
            };
        } else {
            currentNumberBet = null;
        }
    }

    // 👉 LIMPAR
    clearButton.addEventListener('click', () => {
        clearBetsUI();
        console.log("Todas as apostas foram limpas.");
    });

    // 👉 DESFAZER
    undoButton.addEventListener('click', () => {
        const lastBet = bets.pop();
        if (!lastBet) {
            console.log("Não há apostas para desfazer.");
            return;
        }

        if (lastBet.chipElement && lastBet.chipElement.parentNode) {
            lastBet.chipElement.parentNode.removeChild(lastBet.chipElement);
        }

        recalcTotalBet();
        updateCurrentNumberBetFromBets();

        console.log("Aposta desfeita:", lastBet);
    });

    // 👉 REAPOSTAR
    rebetButton.addEventListener('click', () => {
        if (!lastRoundBets.length) {
            alert("Ainda não tens uma ronda anterior para reapostar.");
            return;
        }

        // limpa apostas atuais (mas mantém lastRoundBets)
        clearBetsUI();

        // recriar apostas da ronda anterior
        lastRoundBets.forEach(templateBet => {
            const cell = document.querySelector(`.cell[data-bet="${templateBet.betType}"]`);
            if (!cell) return;

            const chipMarker = document.createElement('div');
            chipMarker.classList.add('bet-chip-marker');
            chipMarker.style.backgroundColor = getChipColor(templateBet.amount);
            cell.appendChild(chipMarker);

            const bet = {
                cell,
                betType: templateBet.betType,
                amount: templateBet.amount,
                number: templateBet.number,
                chipElement: chipMarker
            };

            bets.push(bet);
        });

        recalcTotalBet();
        updateCurrentNumberBetFromBets();

        console.log("Reapostas recriadas:", bets);
    });

    // 👉 X2 (duplicar apostas)
    doubleButton.addEventListener('click', () => {
        if (!bets.length) {
            console.log("Não há apostas para duplicar.");
            return;
        }

        bets.forEach(bet => {
            bet.amount *= 2;
            // se quisesses mostrar o valor na ficha:
            // if (bet.chipElement) bet.chipElement.textContent = bet.amount;
        });

        if (currentNumberBet) {
            currentNumberBet.amount *= 2;
        }

        recalcTotalBet();
        console.log("Apostas duplicadas:", bets, "currentNumberBet:", currentNumberBet);
    });

    // --- 1. Seleção de ficha ---
    chipSelector.addEventListener('click', (event) => {
        if (event.target.classList.contains('chip')) {
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            selectedChipValue = parseFloat(event.target.getAttribute('data-value'));
            console.log(`Selected chip value: ${selectedChipValue}`);
        }
    });

    // ficha 1 como default
    document.querySelector('.chip[data-value="1"]').classList.add('active');

    // --- 2. Colocar apostas ---
    bettingCells.forEach(cell => {
        cell.addEventListener('click', () => {
            const betType = cell.getAttribute('data-bet');

            const numericValue = parseInt(betType, 10);
            const isNumeric = !isNaN(numericValue);

            // criar ficha visual
            const chipMarker = document.createElement('div');
            chipMarker.classList.add('bet-chip-marker');
            chipMarker.style.backgroundColor = getChipColor(selectedChipValue);
            // se quiseres texto na ficha: chipMarker.textContent = selectedChipValue;
            cell.appendChild(chipMarker);

            // guardar aposta
            const bet = {
                cell,
                betType,                  // ex: "17", "par", "1-18"
                amount: selectedChipValue,
                number: isNumeric ? numericValue : null,
                chipElement: chipMarker
            };

            bets.push(bet);

            // se for aposta numérica, é esta que o backend vai usar
            if (isNumeric) {
                currentNumberBet = {
                    amount: bet.amount,
                    number: bet.number
                };
            }

            recalcTotalBet();

            console.log(`Bet placed on: ${betType} with value ${selectedChipValue}`);
            console.log("Estado de bets:", bets);
        });
    });

    // --- 3. Spin + ligação ao backend ---
    const playButton = document.querySelector('.play-btn');

    playButton.addEventListener('click', async () => {
        if (!bets.length) {
            alert("Coloca pelo menos uma aposta primeiro.");
            return;
        }

        const spinDuration = 3 + Math.random() * 2; // 3–5 segundos

        try {
            // enviar todas as apostas para o backend
            const response = await fetch("/api/spin", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ bets: bets }),
            });

            const data = await response.json();
            console.log("Resposta do servidor:", data);

            // descobrir slot pelo índice ou número
            let slotIndex = null;
            if (typeof data.landing_slot_index === "number") {
                slotIndex = data.landing_slot_index;
            } else if (typeof data.landing_number === "number") {
                slotIndex = ROULETTE_ORDER.indexOf(data.landing_number);
            }

            if (slotIndex !== null && slotIndex >= 0) {
                spinBallToSlot(slotIndex, spinDuration);
            } else {
                console.warn("Não foi possível determinar o slot da resposta:", data);
            }

            // atualizar saldo e último ganho
            const currentBalanceRaw = balanceDisplay.textContent.replace("€", "").trim();
            const currentBalance = parseFloat(currentBalanceRaw) || 0;

            const betAmount = Number(data.total_bet) || 0;
            const payoutTotal = Number(data.total_payout) || 0;

            // saldo = saldo - aposta + payout total
            const newBalance = currentBalance - betAmount + payoutTotal;
            balanceDisplay.textContent = `€${newBalance.toFixed(2)}`;

            // último ganho = lucro líquido (se perder, mostra 0)
            const netWin = Math.max(payoutTotal - betAmount, 0);
            lastWinDisplay.textContent = `€${netWin.toFixed(2)}`;

            console.log("DEBUG ganhos:", {
                currentBalance,
                betAmount,
                payoutTotal,
                newBalance,
                netWin,
                results: data.results
            });

            // guardar ronda para REAPOSTAR
            lastRoundBets = bets.map(b => ({
                betType: b.betType,
                amount: b.amount,
                number: b.number
            }));

        } catch (err) {
            console.error("Erro ao comunicar com backend:", err);
            alert("Erro ao comunicar com o servidor.");
        }

        console.log("Spin concluído (com animação da bola).");
    });

});

// ---------- helpers para construir a mesa e roda ----------

function populateNumberCells(container) {
    const layoutRows = [
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
        [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
        [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
    ];

    layoutRows.forEach(row => {
        row.forEach(number => {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.classList.add(RED_NUMBERS.has(number) ? 'red' : 'black');
            cell.dataset.bet = number;
            cell.textContent = number;
            container.appendChild(cell);
        });
    });
}

function setupWheel(wheelInnerRing, markersContainer) {
    const segmentAngle = 360 / ROULETTE_ORDER.length;
    const gradientStops = [];
    const separatorSize = 0.8; // degrees

    markersContainer.innerHTML = '';

    ROULETTE_ORDER.forEach((number, index) => {
        const startAngle = index * segmentAngle;
        const endAngle = startAngle + segmentAngle;

        const color = number === 0 ? '#0b6623' : (RED_NUMBERS.has(number) ? '#b71c1c' : '#1a1a1a');

        const interiorStart = startAngle + separatorSize / 2;
        const interiorEnd = endAngle - separatorSize / 2;

        gradientStops.push(`#1c1c1c ${startAngle}deg ${interiorStart}deg`);
        gradientStops.push(`${color} ${interiorStart}deg ${interiorEnd}deg`);
        gradientStops.push(`#1c1c1c ${interiorEnd}deg ${endAngle}deg`);

        const marker = document.createElement('div');
        marker.classList.add('wheel-number');
        marker.classList.add(number === 0 ? 'green' : (RED_NUMBERS.has(number) ? 'red' : 'black'));
        marker.textContent = number;

        const rotation = startAngle + segmentAngle / 2;

        marker.style.transform =
            `translate(-50%, -50%) rotate(${rotation}deg) translateY(-150px) rotate(${-rotation}deg)`;

        markersContainer.appendChild(marker);
    });

    wheelInnerRing.style.background = `conic-gradient(${gradientStops.join(', ')})`;

    const ball = document.querySelector('.ball');
    if (ball) {
        ball.style.zIndex = 20;
    }
}

function getChipColor(value) {
    if (value === 0.5) return '#880e4f';
    if (value === 1) return '#38761d';
    if (value === 5) return '#0c3498';
    if (value === 10) return '#d58c14';
    if (value === 25) return '#990000';
    if (value === 100) return '#3d3d3d';
    return '#fff';
}
