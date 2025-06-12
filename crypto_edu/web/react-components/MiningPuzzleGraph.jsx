import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './MiningPuzzleGraph.css';

/**
 * Mining Puzzle Graph React Component
 * 
 * This component visualizes the mining puzzle difficulty and
 * provides an interactive demonstration of the mining process.
 */
const MiningPuzzleGraph = ({ width = 800, height = 500 }) => {
  const svgRef = useRef(null);
  const [difficulty, setDifficulty] = useState(4);
  const [nonce, setNonce] = useState(0);
  const [mining, setMining] = useState(false);
  const [hashRate, setHashRate] = useState(100); // Hashes per second
  const [currentHash, setCurrentHash] = useState('');
  const [foundBlock, setFoundBlock] = useState(false);
  const [hashHistory, setHashHistory] = useState([]);
  const [blockData, setBlockData] = useState('Block data: transactions...');
  
  // Calculate target based on difficulty
  const getTarget = (diff) => {
    return '0'.repeat(diff) + 'f'.repeat(64 - diff);
  };
  
  // Calculate hash for given nonce
  const calculateHash = (blockData, nonce) => {
    // In a real implementation, we would use a proper hash function
    // Here we're simulating it with a simple string manipulation for demonstration
    const data = `${blockData}${nonce}`;
    let hash = '';
    
    // Simple hash function for demonstration
    for (let i = 0; i < 64; i++) {
      const charCode = data.charCodeAt(i % data.length) + i + nonce;
      hash += (charCode % 16).toString(16);
    }
    
    return hash;
  };
  
  // Check if hash meets target
  const checkHash = (hash, target) => {
    return BigInt(`0x${hash}`) <= BigInt(`0x${target}`);
  };
  
  // Start mining
  const startMining = () => {
    if (mining) return;
    
    setMining(true);
    setFoundBlock(false);
    setHashHistory([]);
    setNonce(0);
  };
  
  // Stop mining
  const stopMining = () => {
    setMining(false);
  };
  
  // Reset mining
  const resetMining = () => {
    stopMining();
    setNonce(0);
    setCurrentHash('');
    setFoundBlock(false);
    setHashHistory([]);
  };
  
  // Mining effect
  useEffect(() => {
    if (!mining) return;
    
    const target = getTarget(difficulty);
    
    const miningInterval = setInterval(() => {
      // Calculate hash for current nonce
      const hash = calculateHash(blockData, nonce);
      setCurrentHash(hash);
      
      // Add to history (limit to last 10)
      setHashHistory(prev => {
        const newHistory = [...prev, { nonce, hash }];
        if (newHistory.length > 10) {
          return newHistory.slice(1);
        }
        return newHistory;
      });
      
      // Check if hash meets target
      if (checkHash(hash, target)) {
        setFoundBlock(true);
        setMining(false);
      } else {
        setNonce(prev => prev + 1);
      }
    }, 1000 / hashRate);
    
    return () => clearInterval(miningInterval);
  }, [mining, nonce, difficulty, blockData, hashRate]);
  
  // Visualization effect
  useEffect(() => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    // Set up scales
    const xScale = d3.scaleLinear()
      .domain([0, 16])
      .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 16])
      .range([height - 50, 50]);
    
    // Draw axes
    const xAxis = d3.axisBottom(xScale)
      .tickValues(d3.range(0, 17, 4))
      .tickFormat(d => d.toString(16));
    
    const yAxis = d3.axisLeft(yScale)
      .tickValues(d3.range(0, 17, 4))
      .tickFormat(d => d.toString(16));
    
    svg.append('g')
      .attr('transform', `translate(0, ${height - 50})`)
      .call(xAxis);
    
    svg.append('g')
      .attr('transform', `translate(50, 0)`)
      .call(yAxis);
    
    // Add labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 10)
      .attr('text-anchor', 'middle')
      .text('First Hex Digit');
    
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .text('Second Hex Digit');
    
    // Draw target area
    const targetArea = [];
    const target = getTarget(difficulty);
    
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 16; j++) {
        const hexValue = i.toString(16) + j.toString(16);
        if (hexValue <= target.substring(0, 2)) {
          targetArea.push({ x: i, y: j });
        }
      }
    }
    
    svg.selectAll('.target-cell')
      .data(targetArea)
      .enter()
      .append('rect')
      .attr('class', 'target-cell')
      .attr('x', d => xScale(d.x) - 10)
      .attr('y', d => yScale(d.y) - 10)
      .attr('width', 20)
      .attr('height', 20)
      .attr('fill', 'rgba(0, 255, 0, 0.2)')
      .attr('stroke', 'green')
      .attr('stroke-width', 1);
    
    // Draw hash history points
    svg.selectAll('.hash-point')
      .data(hashHistory)
      .enter()
      .append('circle')
      .attr('class', 'hash-point')
      .attr('cx', d => {
        const firstDigit = parseInt(d.hash.charAt(0), 16);
        return xScale(firstDigit);
      })
      .attr('cy', d => {
        const secondDigit = parseInt(d.hash.charAt(1), 16);
        return yScale(secondDigit);
      })
      .attr('r', 5)
      .attr('fill', d => {
        const firstTwoChars = d.hash.substring(0, 2);
        return firstTwoChars <= target.substring(0, 2) ? 'green' : 'red';
      });
    
    // Draw current hash point
    if (currentHash) {
      const firstDigit = parseInt(currentHash.charAt(0), 16);
      const secondDigit = parseInt(currentHash.charAt(1), 16);
      
      svg.append('circle')
        .attr('class', 'current-hash')
        .attr('cx', xScale(firstDigit))
        .attr('cy', yScale(secondDigit))
        .attr('r', 8)
        .attr('fill', foundBlock ? 'green' : 'red')
        .attr('stroke', 'black')
        .attr('stroke-width', 2);
    }
    
    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '18px')
      .attr('font-weight', 'bold')
      .text(`Mining Puzzle (Difficulty: ${difficulty})`);
    
  }, [width, height, difficulty, currentHash, hashHistory, foundBlock]);
  
  return (
    <div className="mining-puzzle-graph">
      <div className="controls">
        <div className="control-group">
          <label>Difficulty:</label>
          <input 
            type="range" 
            min="1" 
            max="8" 
            value={difficulty} 
            onChange={e => setDifficulty(parseInt(e.target.value))}
            disabled={mining}
          />
          <span>{difficulty}</span>
        </div>
        
        <div className="control-group">
          <label>Hash Rate:</label>
          <input 
            type="range" 
            min="1" 
            max="500" 
            value={hashRate} 
            onChange={e => setHashRate(parseInt(e.target.value))}
          />
          <span>{hashRate} H/s</span>
        </div>
        
        <div className="button-group">
          <button onClick={startMining} disabled={mining}>Start Mining</button>
          <button onClick={stopMining} disabled={!mining}>Stop</button>
          <button onClick={resetMining}>Reset</button>
        </div>
      </div>
      
      <svg ref={svgRef} width={width} height={height}></svg>
      
      <div className="mining-info">
        <div className="info-group">
          <label>Block Data:</label>
          <input 
            type="text" 
            value={blockData} 
            onChange={e => setBlockData(e.target.value)}
            disabled={mining}
          />
        </div>
        
        <div className="info-group">
          <label>Current Nonce:</label>
          <span>{nonce}</span>
        </div>
        
        <div className="info-group">
          <label>Current Hash:</label>
          <span className={foundBlock ? 'valid-hash' : ''}>{currentHash}</span>
        </div>
        
        <div className="info-group">
          <label>Target:</label>
          <span>{getTarget(difficulty)}</span>
        </div>
        
        {foundBlock && (
          <div className="success-message">
            Block found! Nonce: {nonce}, Hash: {currentHash}
          </div>
        )}
      </div>
      
      <div className="hash-history">
        <h3>Hash History</h3>
        <table>
          <thead>
            <tr>
              <th>Nonce</th>
              <th>Hash</th>
              <th>Valid</th>
            </tr>
          </thead>
          <tbody>
            {hashHistory.map((item, index) => (
              <tr key={index}>
                <td>{item.nonce}</td>
                <td>{item.hash}</td>
                <td>{checkHash(item.hash, getTarget(difficulty)) ? '✓' : '✗'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MiningPuzzleGraph;