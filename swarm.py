"""
SwarmAI - Multi-Agent Swarm Intelligence System
=============================================

Implements swarm coordination, collective decision-making, and emergent behavior
for autonomous agent collaboration.

Integrates with ai-warroom for orchestration layer.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
import math
import random


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SwarmAI")


# =============================================================================
# Core Data Structures
# =============================================================================

class SwarmBehavior(Enum):
    """Types of swarm behaviors"""
    FLOCKING = "flocking"           # Cohesive group movement
    FORAGING = "foraging"           # Resource discovery and collection
    CONSENSUS = "consensus"         # Collective decision-making
    DIVISION_OF_LABOR = "division"  # Task specialization
    STIGMERGY = "stigmergy"         # Indirect coordination via environment
    EXPLORATION = "exploration"     # Distributed search


class AgentRole(Enum):
    """Specialized roles in the swarm"""
    SCOUT = "scout"           # Exploration and discovery
    WORKER = "worker"         # Task execution
    COORDINATOR = "coordinator"  # Local coordination
    SPECIALIST = "specialist"    # Domain-specific tasks


class PheromoneType(Enum):
    """Types of pheromone trails for stigmergic communication"""
    TASK_COMPLETE = "task_complete"
    RESOURCE_FOUND = "resource_found"
    DANGER = "danger"
    GATHERING_POINT = "gathering"


@dataclass
class Position:
    """2D position for spatial awareness"""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def move_towards(self, target: 'Position', speed: float) -> 'Position':
        """Move towards target at given speed"""
        dx = target.x - self.x
        dy = target.y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < speed:
            return Position(target.x, target.y)
        
        ratio = speed / dist
        return Position(
            self.x + dx * ratio,
            self.y + dy * ratio
        )


@dataclass
class Pheromone:
    """Pheromone trail for stigmergic coordination"""
    type: PheromoneType
    position: Position
    strength: float
    deposited_by: str
    timestamp: float
    decay_rate: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def current_strength(self) -> float:
        """Calculate current strength accounting for decay"""
        age = time.time() - self.timestamp
        return self.strength * math.exp(-self.decay_rate * age)
    
    def is_active(self) -> bool:
        """Check if pheromone is still active (>1% original strength)"""
        return self.current_strength() > 0.01


@dataclass
class SwarmAgent:
    """Individual agent in the swarm"""
    id: str
    role: AgentRole
    position: Position
    velocity: Position = field(default_factory=lambda: Position(0, 0))
    energy: float = 100.0
    task_capacity: int = 5
    current_tasks: List[str] = field(default_factory=list)
    neighbors: Set[str] = field(default_factory=set)
    state: Dict[str, Any] = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=set)
    
    def can_accept_task(self) -> bool:
        """Check if agent can accept more tasks"""
        return len(self.current_tasks) < self.task_capacity and self.energy > 20.0


@dataclass
class SwarmTask:
    """Task to be executed by swarm"""
    id: str
    type: str
    priority: int
    required_agents: int = 1
    required_capabilities: Set[str] = field(default_factory=set)
    assigned_agents: Set[str] = field(default_factory=set)
    position: Optional[Position] = None
    deadline: Optional[float] = None
    result: Optional[Any] = None
    completed: bool = False


# =============================================================================
# Swarm Intelligence Core
# =============================================================================

class SwarmEnvironment:
    """Virtual environment for swarm coordination"""
    
    def __init__(self, width: float = 1000.0, height: float = 1000.0):
        self.width = width
        self.height = height
        self.pheromones: List[Pheromone] = []
        self._lock = asyncio.Lock()
    
    async def deposit_pheromone(self, pheromone: Pheromone):
        """Deposit a pheromone trail"""
        async with self._lock:
            self.pheromones.append(pheromone)
    
    async def get_nearby_pheromones(
        self, 
        position: Position, 
        radius: float,
        pheromone_type: Optional[PheromoneType] = None
    ) -> List[Pheromone]:
        """Get active pheromones within radius"""
        async with self._lock:
            nearby = []
            for pheromone in self.pheromones:
                if not pheromone.is_active():
                    continue
                
                if pheromone_type and pheromone.type != pheromone_type:
                    continue
                
                if position.distance_to(pheromone.position) <= radius:
                    nearby.append(pheromone)
            
            return nearby
    
    async def cleanup_pheromones(self):
        """Remove inactive pheromones"""
        async with self._lock:
            self.pheromones = [p for p in self.pheromones if p.is_active()]


class FlockingBehavior:
    """Implements Reynolds' flocking algorithm"""
    
    def __init__(
        self,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        perception_radius: float = 50.0
    ):
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.perception_radius = perception_radius
    
    def calculate_steering(
        self,
        agent: SwarmAgent,
        neighbors: List[SwarmAgent]
    ) -> Position:
        """Calculate steering vector based on flocking rules"""
        if not neighbors:
            return Position(0, 0)
        
        # Separation: steer away from neighbors
        separation = self._separation(agent, neighbors)
        
        # Alignment: steer towards average heading
        alignment = self._alignment(agent, neighbors)
        
        # Cohesion: steer towards center of mass
        cohesion = self._cohesion(agent, neighbors)
        
        # Combine vectors
        steering = Position(
            separation.x * self.separation_weight +
            alignment.x * self.alignment_weight +
            cohesion.x * self.cohesion_weight,
            
            separation.y * self.separation_weight +
            alignment.y * self.alignment_weight +
            cohesion.y * self.cohesion_weight
        )
        
        return steering
    
    def _separation(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Position:
        """Steer away from neighbors"""
        steer = Position(0, 0)
        for neighbor in neighbors:
            dist = agent.position.distance_to(neighbor.position)
            if dist < self.perception_radius / 2:
                diff_x = agent.position.x - neighbor.position.x
                diff_y = agent.position.y - neighbor.position.y
                steer.x += diff_x / (dist + 0.1)
                steer.y += diff_y / (dist + 0.1)
        return steer
    
    def _alignment(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Position:
        """Align with neighbors' velocity"""
        avg_vel = Position(0, 0)
        for neighbor in neighbors:
            avg_vel.x += neighbor.velocity.x
            avg_vel.y += neighbor.velocity.y
        
        if neighbors:
            avg_vel.x /= len(neighbors)
            avg_vel.y /= len(neighbors)
        
        return Position(
            avg_vel.x - agent.velocity.x,
            avg_vel.y - agent.velocity.y
        )
    
    def _cohesion(self, agent: SwarmAgent, neighbors: List[SwarmAgent]) -> Position:
        """Steer towards center of mass"""
        center = Position(0, 0)
        for neighbor in neighbors:
            center.x += neighbor.position.x
            center.y += neighbor.position.y
        
        if neighbors:
            center.x /= len(neighbors)
            center.y /= len(neighbors)
        
        return Position(
            center.x - agent.position.x,
            center.y - agent.position.y
        )


class ConsensusEngine:
    """Implements consensus decision-making algorithms"""
    
    async def reach_consensus(
        self,
        agents: List[SwarmAgent],
        proposals: Dict[str, Any],
        timeout: float = 10.0
    ) -> Optional[str]:
        """Reach consensus using voting mechanism"""
        if not agents or not proposals:
            return None
        
        # Each agent votes for a proposal
        votes: Dict[str, int] = {prop_id: 0 for prop_id in proposals.keys()}
        
        for agent in agents:
            # Simple voting strategy - can be extended
            vote = self._agent_vote(agent, proposals)
            if vote in votes:
                votes[vote] += 1
        
        # Determine winner (majority)
        max_votes = max(votes.values())
        winners = [prop_id for prop_id, count in votes.items() if count == max_votes]
        
        if len(winners) == 1:
            return winners[0]
        
        # Tie-breaking: random selection
        return random.choice(winners) if winners else None
    
    def _agent_vote(self, agent: SwarmAgent, proposals: Dict[str, Any]) -> str:
        """Agent voting logic - can be customized"""
        # Default: vote for first proposal (can be extended with preferences)
        return list(proposals.keys())[0] if proposals else ""


class TaskAllocator:
    """Allocates tasks to agents using market-based approach"""
    
    def __init__(self):
        self.task_bids: Dict[str, Dict[str, float]] = {}  # task_id -> {agent_id: bid}
    
    async def allocate_tasks(
        self,
        tasks: List[SwarmTask],
        agents: List[SwarmAgent]
    ) -> Dict[str, Set[str]]:
        """Allocate tasks to agents based on bids"""
        allocations: Dict[str, Set[str]] = {}
        
        for task in tasks:
            if task.completed:
                continue
            
            # Get bids from capable agents
            bids = await self._collect_bids(task, agents)
            
            # Select best agents
            selected = self._select_agents(task, bids)
            
            if selected:
                allocations[task.id] = selected
                task.assigned_agents = selected
        
        return allocations
    
    async def _collect_bids(
        self,
        task: SwarmTask,
        agents: List[SwarmAgent]
    ) -> Dict[str, float]:
        """Collect bids from agents for a task"""
        bids = {}
        
        for agent in agents:
            if not agent.can_accept_task():
                continue
            
            # Check capabilities
            if task.required_capabilities and not task.required_capabilities.issubset(agent.capabilities):
                continue
            
            # Calculate bid (higher is better)
            bid = self._calculate_bid(agent, task)
            bids[agent.id] = bid
        
        return bids
    
    def _calculate_bid(self, agent: SwarmAgent, task: SwarmTask) -> float:
        """Calculate agent's bid for a task"""
        bid = agent.energy / 100.0  # Base bid on energy
        
        # Distance penalty if task has location
        if task.position:
            distance = agent.position.distance_to(task.position)
            bid -= distance / 1000.0
        
        # Workload penalty
        bid -= len(agent.current_tasks) / agent.task_capacity
        
        return max(0.0, bid)
    
    def _select_agents(
        self,
        task: SwarmTask,
        bids: Dict[str, float]
    ) -> Set[str]:
        """Select best agents for task"""
        if not bids:
            return set()
        
        # Sort by bid (descending)
        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N agents
        selected = set()
        for agent_id, bid in sorted_bids[:task.required_agents]:
            if bid > 0:
                selected.add(agent_id)
        
        return selected


# =============================================================================
# Main Swarm Coordinator
# =============================================================================

class SwarmCoordinator:
    """Main coordinator for swarm intelligence system"""
    
    def __init__(
        self,
        num_agents: int = 10,
        environment: Optional[SwarmEnvironment] = None
    ):
        self.agents: Dict[str, SwarmAgent] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.environment = environment or SwarmEnvironment()
        
        # Behavior engines
        self.flocking = FlockingBehavior()
        self.consensus = ConsensusEngine()
        self.allocator = TaskAllocator()
        
        # State
        self.running = False
        self._update_interval = 0.1  # 100ms
        
        # Initialize agents
        self._initialize_agents(num_agents)
        
        logger.info(f"SwarmCoordinator initialized with {num_agents} agents")
    
    def _initialize_agents(self, num_agents: int):
        """Initialize swarm agents"""
        roles = [AgentRole.SCOUT, AgentRole.WORKER, AgentRole.COORDINATOR, AgentRole.SPECIALIST]
        
        for i in range(num_agents):
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"
            role = roles[i % len(roles)]
            
            agent = SwarmAgent(
                id=agent_id,
                role=role,
                position=Position(
                    random.uniform(0, self.environment.width),
                    random.uniform(0, self.environment.height)
                ),
                capabilities=self._assign_capabilities(role)
            )
            
            self.agents[agent_id] = agent
    
    def _assign_capabilities(self, role: AgentRole) -> Set[str]:
        """Assign capabilities based on role"""
        base = {"basic"}
        
        if role == AgentRole.SCOUT:
            return base | {"exploration", "sensing"}
        elif role == AgentRole.WORKER:
            return base | {"execution", "processing"}
        elif role == AgentRole.COORDINATOR:
            return base | {"coordination", "communication"}
        elif role == AgentRole.SPECIALIST:
            return base | {"analysis", "optimization"}
        
        return base
    
    async def start(self):
        """Start swarm coordination"""
        if self.running:
            return
        
        self.running = True
        logger.info("Swarm coordination started")
        
        # Start main loop
        asyncio.create_task(self._coordination_loop())
        asyncio.create_task(self._pheromone_cleanup_loop())
    
    async def stop(self):
        """Stop swarm coordination"""
        self.running = False
        logger.info("Swarm coordination stopped")
    
    async def add_task(self, task: SwarmTask):
        """Add task to swarm"""
        self.tasks[task.id] = task
        logger.info(f"Task {task.id} added to swarm")
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        while self.running:
            try:
                # Update agent positions (flocking)
                await self._update_agent_positions()
                
                # Allocate tasks
                await self._allocate_tasks()
                
                # Process stigmergic signals
                await self._process_stigmergy()
                
                # Update agent neighbors
                await self._update_neighbors()
                
                await asyncio.sleep(self._update_interval)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
    
    async def _update_agent_positions(self):
        """Update agent positions using flocking behavior"""
        for agent in self.agents.values():
            # Get neighbors
            neighbors = [
                self.agents[n_id] 
                for n_id in agent.neighbors 
                if n_id in self.agents
            ]
            
            # Calculate steering
            steering = self.flocking.calculate_steering(agent, neighbors)
            
            # Update velocity
            agent.velocity.x += steering.x * 0.1
            agent.velocity.y += steering.y * 0.1
            
            # Limit speed
            speed = math.sqrt(agent.velocity.x**2 + agent.velocity.y**2)
            max_speed = 5.0
            if speed > max_speed:
                agent.velocity.x = (agent.velocity.x / speed) * max_speed
                agent.velocity.y = (agent.velocity.y / speed) * max_speed
            
            # Update position
            agent.position.x += agent.velocity.x
            agent.position.y += agent.velocity.y
            
            # Boundary wrapping
            agent.position.x = agent.position.x % self.environment.width
            agent.position.y = agent.position.y % self.environment.height
    
    async def _allocate_tasks(self):
        """Allocate tasks to agents"""
        pending_tasks = [t for t in self.tasks.values() if not t.completed]
        if not pending_tasks:
            return
        
        agents_list = list(self.agents.values())
        allocations = await self.allocator.allocate_tasks(pending_tasks, agents_list)
        
        # Update agent task lists
        for task_id, agent_ids in allocations.items():
            for agent_id in agent_ids:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    if task_id not in agent.current_tasks:
                        agent.current_tasks.append(task_id)
    
    async def _process_stigmergy(self):
        """Process stigmergic coordination"""
        for agent in self.agents.values():
            # Check for nearby pheromones
            nearby = await self.environment.get_nearby_pheromones(
                agent.position,
                radius=50.0
            )
            
            # React to pheromones (simplified)
            for pheromone in nearby:
                if pheromone.type == PheromoneType.GATHERING_POINT:
                    # Move towards gathering point
                    agent.position = agent.position.move_towards(
                        pheromone.position,
                        speed=2.0
                    )
    
    async def _update_neighbors(self):
        """Update agent neighbor relationships"""
        perception_radius = self.flocking.perception_radius
        
        for agent in self.agents.values():
            agent.neighbors.clear()
            
            for other_id, other in self.agents.items():
                if other_id == agent.id:
                    continue
                
                distance = agent.position.distance_to(other.position)
                if distance < perception_radius:
                    agent.neighbors.add(other_id)
    
    async def _pheromone_cleanup_loop(self):
        """Periodic pheromone cleanup"""
        while self.running:
            await asyncio.sleep(5.0)
            await self.environment.cleanup_pheromones()
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        return {
            "total_agents": len(self.agents),
            "active_tasks": len([t for t in self.tasks.values() if not t.completed]),
            "completed_tasks": len([t for t in self.tasks.values() if t.completed]),
            "active_pheromones": len([p for p in self.environment.pheromones if p.is_active()]),
            "agents_by_role": {
                role.value: len([a for a in self.agents.values() if a.role == role])
                for role in AgentRole
            }
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of SwarmAI"""
    
    # Create swarm coordinator
    swarm = SwarmCoordinator(num_agents=20)
    
    # Start coordination
    await swarm.start()
    
    # Add some tasks
    for i in range(5):
        task = SwarmTask(
            id=f"task_{i}",
            type="data_processing",
            priority=i,
            required_agents=2,
            required_capabilities={"processing"},
            position=Position(
                random.uniform(0, 1000),
                random.uniform(0, 1000)
            )
        )
        await swarm.add_task(task)
    
    # Run for a bit
    logger.info("Swarm running...")
    await asyncio.sleep(5.0)
    
    # Get status
    status = swarm.get_swarm_status()
    logger.info(f"Swarm status: {status}")
    
    # Stop swarm
    await swarm.stop()


if __name__ == "__main__":
    asyncio.run(main())
