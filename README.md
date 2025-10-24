# SwarmAI 🐝

**Multi-Agent Swarm Intelligence System**

SwarmAI implements collective intelligence algorithms for coordinating autonomous agents using biologically-inspired swarm behaviors. Agents self-organize, make collective decisions, and solve complex problems through emergent coordination.

## 🌟 Features

### Core Swarm Behaviors
- **Flocking** - Reynolds' boids algorithm for cohesive group movement
- **Foraging** - Distributed resource discovery and collection
- **Consensus** - Collective decision-making via voting mechanisms
- **Division of Labor** - Dynamic task specialization based on capabilities
- **Stigmergy** - Indirect coordination through environmental modification
- **Exploration** - Coordinated spatial search patterns

### Coordination Mechanisms
- **Pheromone Trails** - Stigmergic communication with decay over time
- **Market-Based Task Allocation** - Agents bid on tasks based on fitness
- **Neighbor-Based Flocking** - Local interactions create global patterns
- **Spatial Awareness** - 2D virtual environment for positioning
- **Role-Based Specialization** - Scout, Worker, Coordinator, Specialist

### Intelligence Patterns
- **Emergent Behavior** - Complex patterns from simple local rules
- **Self-Organization** - No central control, distributed decision-making
- **Scalability** - Performance scales with number of agents
- **Robustness** - Fault-tolerant through redundancy
- **Adaptability** - Dynamic response to changing conditions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 SwarmCoordinator                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Flocking   │  │  Consensus   │  │     Task     │  │
│  │  Behavior   │  │    Engine    │  │  Allocator   │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼─────┐    ┌────▼─────┐    ┌────▼─────┐
   │  Agent   │    │  Agent   │    │  Agent   │
   │  Scout   │    │  Worker  │    │ Specialist│
   └──────────┘    └──────────┘    └──────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                ┌────────▼────────┐
                │  Environment    │
                │  - Pheromones   │
                │  - Spatial Map  │
                └─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from swarm import SwarmCoordinator, SwarmTask, Position

async def main():
    # Create swarm with 20 agents
    swarm = SwarmCoordinator(num_agents=20)
    
    # Start coordination
    await swarm.start()
    
    # Add a task
    task = SwarmTask(
        id="task_1",
        type="data_processing",
        priority=1,
        required_agents=3,
        required_capabilities={"processing"},
        position=Position(500, 500)
    )
    await swarm.add_task(task)
    
    # Monitor swarm
    status = swarm.get_swarm_status()
    print(f"Active agents: {status['total_agents']}")
    print(f"Active tasks: {status['active_tasks']}")
    
    # Stop when done
    await swarm.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 📚 Core Concepts

### Swarm Agents

Each agent has:
- **Role** - Scout, Worker, Coordinator, or Specialist
- **Position** - Spatial location in 2D environment
- **Velocity** - Current movement vector
- **Energy** - Resource level (affects availability)
- **Capabilities** - Set of skills for task matching
- **Neighbors** - Nearby agents for local interaction

```python
from swarm import SwarmAgent, AgentRole, Position

agent = SwarmAgent(
    id="agent_001",
    role=AgentRole.SCOUT,
    position=Position(100, 200),
    capabilities={"exploration", "sensing"}
)
```

### Flocking Behavior

Implements Reynolds' three rules:
1. **Separation** - Avoid crowding neighbors
2. **Alignment** - Steer toward average heading of neighbors
3. **Cohesion** - Move toward average position of neighbors

```python
from swarm import FlockingBehavior

flocking = FlockingBehavior(
    separation_weight=1.5,
    alignment_weight=1.0,
    cohesion_weight=1.0,
    perception_radius=50.0
)

steering = flocking.calculate_steering(agent, neighbors)
```

### Stigmergy (Pheromone Trails)

Agents communicate indirectly by depositing pheromones:

```python
from swarm import Pheromone, PheromoneType, Position

pheromone = Pheromone(
    type=PheromoneType.RESOURCE_FOUND,
    position=Position(300, 400),
    strength=1.0,
    deposited_by="agent_001",
    timestamp=time.time(),
    decay_rate=0.1,
    metadata={"resource_type": "data"}
)

await environment.deposit_pheromone(pheromone)
```

Pheromone strength decays exponentially over time:
```
strength(t) = initial_strength * e^(-decay_rate * age)
```

### Consensus Decision-Making

Agents vote on proposals to reach collective decisions:

```python
from swarm import ConsensusEngine

consensus = ConsensusEngine()

proposals = {
    "option_a": {"cost": 10, "benefit": 20},
    "option_b": {"cost": 15, "benefit": 25}
}

result = await consensus.reach_consensus(agents, proposals)
print(f"Swarm decided: {result}")
```

### Market-Based Task Allocation

Agents bid on tasks based on:
- Energy level
- Current workload
- Distance to task
- Capability match

```python
from swarm import TaskAllocator, SwarmTask

allocator = TaskAllocator()

task = SwarmTask(
    id="task_complex",
    type="analysis",
    priority=5,
    required_agents=5,
    required_capabilities={"analysis", "optimization"}
)

allocations = await allocator.allocate_tasks([task], agents)
```

## 🔧 Configuration

### Environment Parameters

```python
from swarm import SwarmEnvironment

env = SwarmEnvironment(
    width=1000.0,    # Virtual world width
    height=1000.0    # Virtual world height
)
```

### Coordinator Parameters

```python
swarm = SwarmCoordinator(
    num_agents=50,           # Number of agents in swarm
    environment=env          # Custom environment (optional)
)

# Adjust behavior weights
swarm.flocking.separation_weight = 2.0
swarm.flocking.alignment_weight = 1.0
swarm.flocking.cohesion_weight = 1.5
swarm.flocking.perception_radius = 75.0

# Adjust update rate
swarm._update_interval = 0.05  # 50ms updates
```

## 📊 Monitoring & Metrics

### Swarm Status

```python
status = swarm.get_swarm_status()

# Returns:
{
    "total_agents": 20,
    "active_tasks": 3,
    "completed_tasks": 7,
    "active_pheromones": 12,
    "agents_by_role": {
        "scout": 5,
        "worker": 10,
        "coordinator": 3,
        "specialist": 2
    }
}
```

### Individual Agent Status

```python
agent = swarm.agents["agent_001"]

print(f"Position: ({agent.position.x}, {agent.position.y})")
print(f"Energy: {agent.energy}")
print(f"Tasks: {len(agent.current_tasks)}")
print(f"Neighbors: {len(agent.neighbors)}")
```

## 🎯 Use Cases

### Distributed Task Processing
Assign computational tasks to agent swarm with automatic load balancing:
```python
# Add 100 tasks, swarm automatically distributes
for i in range(100):
    task = SwarmTask(id=f"task_{i}", type="compute", priority=i % 5)
    await swarm.add_task(task)
```

### Resource Discovery
Agents explore environment to find and collect resources:
```python
# Scouts explore, workers collect, coordinators organize
# Pheromones mark discovered resources
```

### Collective Problem Solving
Reach consensus on best solution through voting:
```python
solutions = {
    "approach_a": {...},
    "approach_b": {...},
    "approach_c": {...}
}
best = await swarm.consensus.reach_consensus(swarm.agents.values(), solutions)
```

### Adaptive Coordination
Swarm adapts to changing conditions through emergent behavior:
- Agents redistribute when workload changes
- Flocking maintains cohesion during movement
- Stigmergy enables indirect coordination

## 🔌 Integration with ai-warroom

SwarmAI designed to integrate with [ai-warroom](https://github.com/BossX429/ai-warroom) orchestration:

```python
# Register swarm agents with Warroom
from warroom import Warroom

warroom = Warroom()
swarm = SwarmCoordinator(num_agents=20)

# Register each swarm agent
for agent_id, agent in swarm.agents.items():
    await warroom.register_agent(
        agent_id=agent_id,
        capabilities=list(agent.capabilities),
        metadata={
            "role": agent.role.value,
            "swarm_coordinator": "swarm_001"
        }
    )

# Warroom assigns tasks, swarm coordinates execution
```

## 🧪 Examples

See `examples/` directory:
- `simple_swarm.py` - Basic swarm creation and monitoring
- `task_allocation.py` - Market-based task distribution
- `pheromone_trails.py` - Stigmergic coordination demo
- `consensus_voting.py` - Collective decision-making
- `warroom_integration.py` - Integration with ai-warroom

## 🏛️ Algorithm Details

### Flocking (Reynolds 1987)
Based on Craig Reynolds' seminal work on emergent flocking behavior. Each agent follows three simple rules creating complex group dynamics.

### Stigmergy (Grassé 1959)
Inspired by termite construction and ant foraging. Agents modify environment through pheromone deposition, creating indirect communication channel.

### Market-Based Allocation (Smith 1980)
Task allocation through bidding mechanism. Agents bid based on fitness, winning bids assigned tasks.

### Consensus Algorithms (Olfati-Saber 2007)
Distributed consensus through local voting and information propagation.

## 🔬 Performance

- **Agent Update Rate**: 100Hz (configurable)
- **Pheromone Cleanup**: Every 5 seconds
- **Task Allocation**: O(n*m) where n=agents, m=tasks
- **Neighbor Detection**: O(n²) with spatial optimization
- **Scalability**: Tested up to 1000 agents

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Linting
```bash
flake8 swarm.py
black swarm.py
mypy swarm.py
```

### Performance Profiling
```bash
python -m cProfile -o swarm.prof swarm.py
```

## 📖 References

- Reynolds, C. W. (1987). "Flocks, herds and schools: A distributed behavioral model"
- Dorigo, M., & Birattari, M. (2010). "Ant colony optimization"
- Bonabeau, E., et al. (1999). "Swarm Intelligence: From Natural to Artificial Systems"
- Olfati-Saber, R., & Murray, R. M. (2004). "Consensus problems in networks of agents"

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New swarm behaviors (ant colony optimization, particle swarm, etc.)
- Performance optimizations (spatial hashing, quadtrees)
- Additional consensus algorithms
- Integration with other AI frameworks
- Visualization tools

## 📬 Contact

- GitHub Issues: Bug reports and feature requests
- Discussions: Architecture and design questions

---

**Built for emergent intelligence through collective behavior** 🐝✨
