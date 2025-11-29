// script.js

const RED_NUMBERS = new Set([1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]);
const ROULETTE_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26];

document.addEventListener('DOMContentLoaded', () => {
    const chipSelector = document.querySelector('.chip-selector');
    const totalBetDisplay = document.getElementById('total-bet-display');
    const numberCellsContainer = document.querySelector('.number-cells');
    const wheelInnerRing = document.querySelector('.wheel-inner-ring');
    const wheelNumberMarkers = document.querySelector('.wheel-number-markers');

    populateNumberCells(numberCellsContainer);
    setupWheel(wheelInnerRing, wheelNumberMarkers);

    const bettingCells = document.querySelectorAll('#betting-grid .cell');
    
    let selectedChipValue = 1; // Default chip value
    let totalBet = 50.00;

    // --- 1. Chip Selection ---
    chipSelector.addEventListener('click', (event) => {
        if (event.target.classList.contains('chip')) {
            // Remove 'active' class from all chips
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            
            // Set new active chip and value
            event.target.classList.add('active');
            selectedChipValue = parseFloat(event.target.getAttribute('data-value'));
            console.log(`Selected chip value: ${selectedChipValue}`);
        }
    });

    // Initialize the default selected chip
    document.querySelector('.chip[data-value="1"]').classList.add('active');


    // --- 2. Placing Bets ---
    bettingCells.forEach(cell => {
        cell.addEventListener('click', () => {
            const betType = cell.getAttribute('data-bet');
            
            // For a basic UI, we just simulate the placement of a chip
            const chipMarker = document.createElement('div');
            chipMarker.classList.add('bet-chip-marker');
            chipMarker.textContent = selectedChipValue;
            chipMarker.style.backgroundColor = getChipColor(selectedChipValue);

            // Simple placement at the center of the cell
            cell.appendChild(chipMarker);

            // Update Total Bet
            totalBet += selectedChipValue;
            totalBetDisplay.textContent = totalBet.toFixed(2);

            console.log(`Bet placed on: ${betType} with value ${selectedChipValue}`);
        });
    });

    // --- 3. Wheel Spin Simulation (Optional) ---
    const playButton = document.querySelector('.play-btn');

    playButton.addEventListener('click', () => {
        // Stop any current animation and restart it for a new spin effect
        wheelInnerRing.style.animation = 'none';
        void wheelInnerRing.offsetWidth; // Trigger reflow
        
        // Simulate a new spin with a random duration
        const spinDuration = 3 + Math.random() * 2; // 3 to 5 seconds
        
        wheelInnerRing.style.animation = `spin-wheel ${spinDuration}s cubic-bezier(0.1, 0.7, 1.0, 0.1) forwards`;
        
        setTimeout(() => {
            console.log('Spin complete. Ready for next bet.');
            wheelInnerRing.style.animation = 'spin-wheel 5s linear infinite';
        }, spinDuration * 1000);
    });

});

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

    // Clear markers
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
