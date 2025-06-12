import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './MerkleTreeViewer.css';

/**
 * Interactive Merkle Tree Viewer React Component
 * 
 * This component visualizes a Merkle tree and allows interactive
 * exploration of proofs and tree construction.
 */
const MerkleTreeViewer = ({ treeData, width = 800, height = 600 }) => {
  const svgRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [proofPath, setProofPath] = useState([]);
  const [animationStep, setAnimationStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  
  // Process tree data
  const processTreeData = () => {
    if (!treeData || !treeData.tree || treeData.tree.length === 0) {
      return { nodes: [], links: [] };
    }
    
    const nodes = [];
    const links = [];
    
    // Process each level
    treeData.tree.forEach(level => {
      level.nodes.forEach(node => {
        nodes.push({
          id: `${level.level}_${node.index}`,
          level: level.level,
          index: node.index,
          hash: node.hash,
          x: 0,  // Will be calculated by d3
          y: 0   // Will be calculated by d3
        });
        
        // Add links to children
        node.children.forEach(child => {
          links.push({
            source: `${level.level}_${node.index}`,
            target: `${child.level}_${child.index}`,
            value: 1
          });
        });
      });
    });
    
    return { nodes, links };
  };
  
  // Generate proof path for a leaf node
  const generateProof = (leafIndex) => {
    if (!treeData || !treeData.tree || treeData.tree.length === 0) {
      return [];
    }
    
    const proof = [];
    let currentLevel = treeData.tree.length - 1;  // Start at leaf level
    let currentIndex = leafIndex;
    
    while (currentLevel > 0) {
      // Find sibling
      const isRightChild = currentIndex % 2 === 1;
      const siblingIndex = isRightChild ? currentIndex - 1 : currentIndex + 1;
      
      // Check if sibling exists
      const levelNodes = treeData.tree[currentLevel].nodes;
      if (siblingIndex >= 0 && siblingIndex < levelNodes.length) {
        proof.push({
          level: currentLevel,
          index: siblingIndex,
          hash: levelNodes[siblingIndex].hash,
          position: isRightChild ? 'left' : 'right'
        });
      }
      
      // Move to parent
      currentIndex = Math.floor(currentIndex / 2);
      currentLevel--;
    }
    
    return proof;
  };
  
  // Render the tree visualization
  useEffect(() => {
    if (!svgRef.current || !treeData) return;
    
    const { nodes, links } = processTreeData();
    if (nodes.length === 0) return;
    
    // Clear previous SVG content
    d3.select(svgRef.current).selectAll('*').remove();
    
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    // Create hierarchical layout
    const maxLevel = Math.max(...nodes.map(n => n.level));
    const levelHeight = height / (maxLevel + 1);
    
    // Calculate x positions based on index within level
    const levelWidths = {};
    nodes.forEach(node => {
      if (!levelWidths[node.level]) {
        levelWidths[node.level] = 0;
      }
      levelWidths[node.level]++;
    });
    
    nodes.forEach(node => {
      const levelWidth = levelWidths[node.level];
      const xStep = width / (levelWidth + 1);
      node.x = (node.index + 1) * xStep;
      node.y = node.level * levelHeight + 50;
    });
    
    // Draw links
    svg.selectAll('.link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('x1', d => {
        const sourceNode = nodes.find(n => n.id === d.source);
        return sourceNode ? sourceNode.x : 0;
      })
      .attr('y1', d => {
        const sourceNode = nodes.find(n => n.id === d.source);
        return sourceNode ? sourceNode.y : 0;
      })
      .attr('x2', d => {
        const targetNode = nodes.find(n => n.id === d.target);
        return targetNode ? targetNode.x : 0;
      })
      .attr('y2', d => {
        const targetNode = nodes.find(n => n.id === d.target);
        return targetNode ? targetNode.y : 0;
      })
      .attr('stroke', '#999')
      .attr('stroke-width', 1);
    
    // Draw nodes
    const nodeElements = svg.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.x}, ${d.y})`)
      .on('click', (event, d) => {
        setSelectedNode(d);
        
        // If it's a leaf node, generate proof
        if (d.level === maxLevel) {
          const proof = generateProof(d.index);
          setProofPath(proof);
        }
      });
    
    // Add circles for nodes
    nodeElements.append('circle')
      .attr('r', 20)
      .attr('fill', d => {
        // Color based on level
        if (d.level === 0) return '#ff7f0e';  // Root
        if (d.level === maxLevel) return '#1f77b4';  // Leaves
        return '#2ca02c';  // Internal nodes
      })
      .attr('stroke', d => {
        // Highlight selected node
        if (selectedNode && d.id === selectedNode.id) return '#d62728';
        
        // Highlight proof path
        if (proofPath.some(p => `${p.level}_${p.index}` === d.id)) return '#9467bd';
        
        return '#fff';
      })
      .attr('stroke-width', 2);
    
    // Add labels
    nodeElements.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .text(d => d.hash.substring(0, 4) + '...')
      .attr('fill', '#fff');
    
    // Add tooltips
    nodeElements.append('title')
      .text(d => `Level: ${d.level}\nIndex: ${d.index}\nHash: ${d.hash}`);
    
  }, [treeData, width, height, selectedNode, proofPath, animationStep]);
  
  // Animation effect
  useEffect(() => {
    if (!isAnimating) return;
    
    const maxSteps = treeData?.tree?.length || 0;
    
    const timer = setTimeout(() => {
      if (animationStep < maxSteps - 1) {
        setAnimationStep(animationStep + 1);
      } else {
        setIsAnimating(false);
      }
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [isAnimating, animationStep, treeData]);
  
  // Start animation
  const startAnimation = () => {
    setAnimationStep(0);
    setIsAnimating(true);
  };
  
  return (
    <div className="merkle-tree-viewer">
      <div className="controls">
        <button onClick={startAnimation} disabled={isAnimating}>
          {isAnimating ? 'Animating...' : 'Animate Construction'}
        </button>
        <button onClick={() => setProofPath([])}>Clear Proof</button>
      </div>
      
      <svg ref={svgRef}></svg>
      
      {selectedNode && (
        <div className="node-details">
          <h3>Node Details</h3>
          <p><strong>Level:</strong> {selectedNode.level}</p>
          <p><strong>Index:</strong> {selectedNode.index}</p>
          <p><strong>Hash:</strong> {selectedNode.hash}</p>
        </div>
      )}
      
      {proofPath.length > 0 && (
        <div className="proof-details">
          <h3>Merkle Proof</h3>
          <ul>
            {proofPath.map((step, i) => (
              <li key={i}>
                Step {i+1}: {step.position} sibling hash: {step.hash.substring(0, 8)}...
              </li>
            ))}
          </ul>
          <p>
            <strong>Verification:</strong> 
            {proofPath.length > 0 ? ' Proof can be verified against root hash' : ' No proof selected'}
          </p>
        </div>
      )}
    </div>
  );
};

export default MerkleTreeViewer;