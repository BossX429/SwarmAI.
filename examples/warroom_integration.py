"""
Example: Integration between SwarmAI and ai-warroom
Demonstrates how swarm agents work with orchestration layer
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm import (
    SwarmCoordinator,
    SwarmTask,
    Position,
    AgentRole
)


class WarroomSwarmBridge:
    """Bridge between ai-warroom orchestration and SwarmAI coordination"""
    
    def __init__(self, swarm: SwarmCoordinator):
        self.swarm = swarm
        self.task_results = {}
    
    async def register_swarm_with_warroom(self):
        """Register all swarm agents with warroom (simulated)"""
        print("\n=== Registering Swarm Agents with Warroom ===")
        
        for agent_id, agent in self.swarm.agents.items():
            # In real integration, this would call Warroom.register_agent()
            print(f"Registered: {agent_id}")
            print(f"  Role: {agent.role.value}")
            print(f"  Capabilities: {agent.capabilities}")
            print(f"  Position: ({agent.position.x:.1f}, {agent.position.y:.1f})")
        
        print(f"\nTotal agents registered: {len(self.swarm.agents)}")
    
    async def receive_task_from_warroom(self, task_id: str, task_type: str):
        """Receive task from warroom and distribute to swarm"""
        print(f"\n=== Received Task from Warroom: {task_id} ===")
        
        # Create swarm task
        task = SwarmTask(
            id=task_id,
            type=task_type,
            priority=1,
            required_agents=3,
            required_capabilities={"processing", "analysis"},
            position=Position(500, 500)
        )
        
        # Add to swarm
        await self.swarm.add_task(task)
        print(f"Task {task_id} distributed to swarm")
        
        return task
    
    async def monitor_task_execution(self, task_id: str, duration: float = 5.0):
        """Monitor swarm executing a task"""
        print(f"\n=== Monitoring Task Execution: {task_id} ===")
        
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < duration:
            # Get swarm status
            status = self.swarm.get_swarm_status()
            
            # Find assigned agents
            task = self.swarm.tasks.get(task_id)
            if task:
                print(f"\rTask {task_id}: {len(task.assigned_agents)} agents assigned", end="")
            
            await asyncio.sleep(0.5)
        
        print(f"\n\nTask execution monitoring complete")
        
        # Return result to warroom
        if task and task.assigned_agents:
            result = {
                "task_id": task_id,
                "assigned_agents": list(task.assigned_agents),
                "completion_status": "success"
            }
            self.task_results[task_id] = result
            print(f"Result ready to return to warroom: {result}")


async def demo_integration():
    """Demonstrate SwarmAI + ai-warroom integration"""
    
    print("╔════════════════════════════════════════════╗")
    print("║   SwarmAI + ai-warroom Integration Demo   ║")
    print("╚════════════════════════════════════════════╝")
    
    # Create swarm coordinator
    print("\n1. Creating swarm coordinator with 15 agents...")
    swarm = SwarmCoordinator(num_agents=15)
    
    # Start swarm
    print("2. Starting swarm coordination...")
    await swarm.start()
    await asyncio.sleep(1)
    
    # Create bridge
    print("3. Creating warroom-swarm bridge...")
    bridge = WarroomSwarmBridge(swarm)
    
    # Register agents with warroom
    await bridge.register_swarm_with_warroom()
    await asyncio.sleep(1)
    
    # Simulate warroom sending tasks
    print("\n4. Warroom sending tasks to swarm...")
    task1 = await bridge.receive_task_from_warroom("warroom_task_001", "data_analysis")
    await asyncio.sleep(0.5)
    
    task2 = await bridge.receive_task_from_warroom("warroom_task_002", "optimization")
    await asyncio.sleep(0.5)
    
    task3 = await bridge.receive_task_from_warroom("warroom_task_003", "processing")
    
    # Monitor execution
    print("\n5. Monitoring swarm task execution...")
    await bridge.monitor_task_execution("warroom_task_001", duration=3.0)
    
    # Show final swarm status
    print("\n6. Final swarm status:")
    status = swarm.get_swarm_status()
    print(f"   Total agents: {status['total_agents']}")
    print(f"   Active tasks: {status['active_tasks']}")
    print(f"   Completed tasks: {status['completed_tasks']}")
    print(f"   Active pheromones: {status['active_pheromones']}")
    print(f"   Agents by role:")
    for role, count in status['agents_by_role'].items():
        print(f"     {role}: {count}")
    
    # Show results
    print("\n7. Results collected for warroom:")
    for task_id, result in bridge.task_results.items():
        print(f"   {task_id}: {result['completion_status']}")
        print(f"     Agents: {', '.join(result['assigned_agents'][:3])}...")
    
    # Stop swarm
    print("\n8. Stopping swarm coordination...")
    await swarm.stop()
    
    print("\n✓ Integration demo complete!")
    print("\nKey Takeaways:")
    print("  • Warroom provides high-level orchestration")
    print("  • SwarmAI handles collective intelligence and coordination")
    print("  • Tasks flow from warroom → swarm → agents")
    print("  • Results flow from agents → swarm → warroom")
    print("  • Swarm self-organizes task distribution")
    print("  • Emergent behavior enables efficient execution")


if __name__ == "__main__":
    asyncio.run(demo_integration())
