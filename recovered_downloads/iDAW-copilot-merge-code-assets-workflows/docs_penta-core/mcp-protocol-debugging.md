# MCP Protocol Implementation & Autonomous Debugging Strategy

A comprehensive protocol specification and debugging strategy for autonomous operation when the human operator is not active.

---

## Table of Contents

1. [MCP Protocol Specification](#1-mcp-protocol-specification)
2. [Autonomous Operation Modes](#2-autonomous-operation-modes)
3. [Debugging Infrastructure](#3-debugging-infrastructure)
4. [Self-Healing Mechanisms](#4-self-healing-mechanisms)
5. [Monitoring & Alerting](#5-monitoring--alerting)
6. [Runbooks for Common Issues](#6-runbooks-for-common-issues)
7. [Implementation Checklist](#7-implementation-checklist)

---

## 1. MCP Protocol Specification

### 1.1 Message Format

All MCP messages follow this standardized format:

```json
{
  "version": "1.0.0",
  "message_id": "uuid-v4",
  "timestamp": "ISO-8601",
  "sender": {
    "agent_id": "string",
    "role": "planner|researcher|builder|tester|auditor",
    "session_id": "uuid-v4"
  },
  "receiver": {
    "agent_id": "string|broadcast",
    "role": "string|*"
  },
  "intent": "string",
  "payload": {},
  "metadata": {
    "priority": 1-10,
    "ttl_seconds": 3600,
    "idempotency_key": "uuid-v4",
    "correlation_id": "uuid-v4",
    "trace_id": "uuid-v4",
    "parent_message_id": "uuid-v4|null"
  }
}
```

### 1.2 Intent Categories

| Category | Intents | Description |
|----------|---------|-------------|
| **Lifecycle** | `init`, `ready`, `pause`, `resume`, `shutdown` | Agent lifecycle management |
| **Task** | `task.create`, `task.assign`, `task.start`, `task.complete`, `task.fail` | Task management |
| **Query** | `query.status`, `query.capability`, `query.resource` | Information requests |
| **Tool** | `tool.invoke`, `tool.result`, `tool.error` | Tool execution |
| **Sync** | `sync.checkpoint`, `sync.restore`, `sync.conflict` | State synchronization |
| **Debug** | `debug.trace`, `debug.dump`, `debug.breakpoint` | Debugging operations |
| **Health** | `health.ping`, `health.pong`, `health.report` | Health monitoring |

### 1.3 Tool Invocation Protocol

```json
// Request
{
  "intent": "tool.invoke",
  "payload": {
    "tool_name": "file_write",
    "tool_version": "1.0.0",
    "inputs": {
      "path": "/path/to/file",
      "content": "..."
    },
    "options": {
      "timeout_ms": 30000,
      "retry_count": 3,
      "dry_run": false
    }
  }
}

// Response
{
  "intent": "tool.result",
  "payload": {
    "tool_name": "file_write",
    "status": "success|error|timeout|cancelled",
    "outputs": {},
    "duration_ms": 1234,
    "resource_usage": {
      "cpu_ms": 100,
      "memory_mb": 50,
      "io_operations": 2
    }
  }
}
```

### 1.4 Error Response Format

```json
{
  "intent": "tool.error",
  "payload": {
    "error_code": "ERR_FILE_NOT_FOUND",
    "error_category": "recoverable|fatal|transient",
    "message": "Human-readable error message",
    "details": {
      "path": "/path/that/failed",
      "attempted_at": "ISO-8601"
    },
    "recovery_hints": [
      "Check if the file exists",
      "Verify permissions",
      "Try with absolute path"
    ],
    "stack_trace": "optional string for debugging"
  }
}
```

---

## 2. Autonomous Operation Modes

### 2.1 Operating States

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS OPERATION FSM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    human_active=false    ┌──────────────┐        │
│   │  ACTIVE  │ ─────────────────────▶   │  AUTONOMOUS  │        │
│   └──────────┘                          └──────────────┘        │
│        │                                       │                 │
│        │ error                          error  │                 │
│        ▼                                       ▼                 │
│   ┌──────────┐    auto_recover          ┌──────────────┐        │
│   │  PAUSED  │ ◀─────────────────────── │ SAFE_MODE    │        │
│   └──────────┘                          └──────────────┘        │
│        │                                       │                 │
│        │ human_resume                   human_required           │
│        ▼                                       ▼                 │
│   ┌──────────┐                          ┌──────────────┐        │
│   │  ACTIVE  │                          │ WAITING_HUMAN│        │
│   └──────────┘                          └──────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Autonomous Mode Configuration

```yaml
# autonomous_config.yaml
autonomous_mode:
  enabled: true
  activation_trigger: "human_inactive_for_minutes: 5"
  
  allowed_operations:
    - read_files
    - write_docs
    - run_tests
    - generate_reports
    - fix_lint_errors
    - update_dependencies_minor
    
  forbidden_operations:
    - delete_files
    - push_to_main
    - modify_secrets
    - external_api_calls_destructive
    - database_schema_changes
    
  resource_limits:
    max_file_writes_per_hour: 100
    max_api_calls_per_minute: 30
    max_cpu_seconds: 3600
    max_memory_mb: 4096
    max_disk_writes_mb: 1000
    
  safety_constraints:
    require_tests_pass: true
    require_lint_pass: true
    max_consecutive_failures: 5
    rollback_on_test_failure: true
    checkpoint_every_n_operations: 10
```

### 2.3 Task Queue for Autonomous Work

```json
{
  "queue_name": "autonomous_work",
  "tasks": [
    {
      "task_id": "uuid",
      "priority": 1,
      "type": "continuous|scheduled|triggered",
      "schedule": "*/15 * * * *",
      "action": {
        "intent": "tool.invoke",
        "tool": "run_tests",
        "params": {}
      },
      "on_success": "log_and_continue",
      "on_failure": "retry_then_alert",
      "max_retries": 3,
      "timeout_seconds": 300
    }
  ]
}
```

---

## 3. Debugging Infrastructure

### 3.1 Trace Levels

| Level | Name | Description | Use Case |
|-------|------|-------------|----------|
| 0 | OFF | No tracing | Production (minimal) |
| 1 | ERROR | Errors only | Production (default) |
| 2 | WARN | Warnings + errors | Production (verbose) |
| 3 | INFO | Key events | Staging |
| 4 | DEBUG | Detailed operations | Development |
| 5 | TRACE | Everything including data | Deep debugging |

### 3.2 Structured Logging Format

```json
{
  "timestamp": "2024-01-15T14:30:00.123Z",
  "level": "INFO",
  "logger": "mcp.agent.builder",
  "message": "Task completed successfully",
  "context": {
    "run_id": "uuid",
    "task_id": "uuid",
    "agent_id": "builder-01",
    "correlation_id": "uuid",
    "trace_id": "uuid"
  },
  "metrics": {
    "duration_ms": 1234,
    "retries": 0,
    "memory_mb": 256
  },
  "tags": ["task", "success", "autonomous"]
}
```

### 3.3 Debug Commands

```python
# Debug command interface
class DebugCommands:
    
    def dump_state(self, agent_id: str = "*") -> dict:
        """Dump current state of one or all agents"""
        pass
    
    def set_breakpoint(self, 
                       condition: str,
                       action: str = "pause") -> str:
        """
        Set conditional breakpoint
        condition: "error_count > 3", "task.type == 'file_write'"
        action: "pause", "log", "alert", "snapshot"
        """
        pass
    
    def trace_message(self, message_id: str) -> list:
        """Trace a message through the entire system"""
        pass
    
    def replay_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Replay operations from a specific checkpoint"""
        pass
    
    def simulate_failure(self, 
                         failure_type: str,
                         target: str) -> dict:
        """Inject failure for testing recovery"""
        pass
    
    def get_metrics_snapshot(self) -> dict:
        """Get current metrics for all agents"""
        pass
```

### 3.4 Checkpoint System

```python
@dataclass
class Checkpoint:
    checkpoint_id: str
    timestamp: datetime
    agent_states: dict[str, AgentState]
    task_queue: list[Task]
    pending_messages: list[Message]
    file_hashes: dict[str, str]  # path -> sha256
    metadata: dict
    
class CheckpointManager:
    def create_checkpoint(self, 
                          trigger: str = "manual") -> Checkpoint:
        """Create a system checkpoint"""
        pass
    
    def restore_checkpoint(self, 
                           checkpoint_id: str,
                           partial: bool = False) -> bool:
        """Restore system to checkpoint state"""
        pass
    
    def diff_checkpoints(self,
                         checkpoint_a: str,
                         checkpoint_b: str) -> CheckpointDiff:
        """Compare two checkpoints"""
        pass
    
    def cleanup_old_checkpoints(self,
                                 keep_last_n: int = 10,
                                 keep_days: int = 7) -> int:
        """Cleanup old checkpoints, return count deleted"""
        pass
```

---

## 4. Self-Healing Mechanisms

### 4.1 Recovery Actions

```yaml
recovery_policies:
  
  transient_error:
    description: "Network timeout, rate limit, temporary unavailable"
    actions:
      - wait_exponential_backoff:
          initial_ms: 1000
          max_ms: 60000
          multiplier: 2
      - retry:
          max_attempts: 5
      - if_still_failing:
          escalate_to: safe_mode
  
  tool_failure:
    description: "A tool returns an error"
    actions:
      - log_error:
          level: WARN
      - check_idempotency:
          if_idempotent: retry
          if_not: skip_and_report
      - after_n_failures: 3
          switch_to_fallback_tool: true
  
  agent_unresponsive:
    description: "Agent stops responding to health checks"
    actions:
      - send_health_ping:
          timeout_ms: 5000
          retries: 3
      - if_no_response:
          - log_error
          - restart_agent
          - restore_from_last_checkpoint
      - if_restart_fails:
          - mark_agent_dead
          - redistribute_tasks
          - alert_human
  
  consistency_error:
    description: "State inconsistency detected"
    actions:
      - pause_all_operations
      - create_diagnostic_snapshot
      - attempt_auto_repair:
          strategies:
            - rebuild_from_source_of_truth
            - replay_from_last_good_checkpoint
            - rollback_to_safe_state
      - if_repair_fails:
          - enter_safe_mode
          - require_human_intervention
```

### 4.2 Circuit Breaker Pattern

```python
class CircuitBreaker:
    """
    Prevents cascading failures by stopping calls to failing services
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 reset_timeout_seconds: int = 60,
                 half_open_max_calls: int = 3):
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_try_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError("Circuit is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = "CLOSED"
                self.failure_count = 0
        
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

### 4.3 Automatic Rollback

```python
class RollbackManager:
    """
    Manages automatic rollbacks on failure
    """
    
    def __init__(self):
        self.operations_stack = []
        self.rollback_handlers = {}
        
    def register_rollback(self, 
                          operation_type: str,
                          rollback_func: Callable):
        """Register a rollback handler for an operation type"""
        self.rollback_handlers[operation_type] = rollback_func
        
    @contextmanager
    def transaction(self, name: str):
        """Execute operations in a transaction with auto-rollback"""
        checkpoint_id = self.checkpoint_manager.create_checkpoint(
            trigger=f"transaction:{name}"
        )
        try:
            yield
        except Exception as e:
            self.rollback_to(checkpoint_id)
            raise TransactionRolledBack(
                f"Transaction {name} rolled back due to: {e}"
            )
    
    def rollback_to(self, checkpoint_id: str):
        """Execute rollback to a specific checkpoint"""
        while self.operations_stack:
            op = self.operations_stack.pop()
            if op.checkpoint_id == checkpoint_id:
                break
            handler = self.rollback_handlers.get(op.type)
            if handler:
                handler(op)
```

---

## 5. Monitoring & Alerting

### 5.1 Metrics Collection

```yaml
metrics:
  # Agent metrics
  agent_health:
    - name: agent_status
      type: gauge
      labels: [agent_id, role]
    - name: agent_uptime_seconds
      type: counter
      labels: [agent_id]
    - name: agent_memory_bytes
      type: gauge
      labels: [agent_id]
    - name: agent_cpu_usage_percent
      type: gauge
      labels: [agent_id]
      
  # Task metrics
  task_metrics:
    - name: tasks_total
      type: counter
      labels: [status, type, agent]
    - name: task_duration_seconds
      type: histogram
      labels: [type, agent]
      buckets: [0.1, 0.5, 1, 5, 10, 30, 60, 300]
    - name: task_queue_depth
      type: gauge
      labels: [priority, agent]
      
  # Tool metrics
  tool_metrics:
    - name: tool_calls_total
      type: counter
      labels: [tool, status]
    - name: tool_duration_seconds
      type: histogram
      labels: [tool]
    - name: tool_errors_total
      type: counter
      labels: [tool, error_code]
      
  # System metrics
  system_metrics:
    - name: message_queue_depth
      type: gauge
    - name: checkpoint_count
      type: gauge
    - name: autonomous_mode_active
      type: gauge
      labels: [agent_id]
```

### 5.2 Alert Rules

```yaml
alerts:
  
  critical:
    - name: AllAgentsDown
      condition: sum(agent_status{status="healthy"}) == 0
      for: 1m
      action: page_oncall
      
    - name: ConsecutiveTaskFailures
      condition: tasks_failed_consecutive > 10
      for: 0s
      action: pause_and_alert
      
    - name: SafeModeEntered
      condition: agent_mode == "safe_mode"
      for: 0s
      action: alert_human_required
      
  warning:
    - name: HighErrorRate
      condition: rate(tool_errors_total[5m]) > 0.1
      for: 5m
      action: log_and_alert
      
    - name: SlowTaskExecution
      condition: histogram_quantile(0.95, task_duration_seconds) > 60
      for: 10m
      action: log_warning
      
    - name: HighMemoryUsage
      condition: agent_memory_bytes > 3000000000  # 3GB
      for: 5m
      action: trigger_gc_and_alert
      
  info:
    - name: LongQueueTime
      condition: task_queue_wait_seconds > 300
      for: 15m
      action: log_info
      
    - name: CheckpointCreated
      condition: checkpoint_created == 1
      action: log_info
```

### 5.3 Health Check System

```python
class HealthCheckSystem:
    """
    Comprehensive health monitoring
    """
    
    def __init__(self):
        self.checks = {}
        self.last_results = {}
        
    def register_check(self,
                       name: str,
                       check_func: Callable,
                       interval_seconds: int = 30,
                       timeout_seconds: int = 10,
                       critical: bool = False):
        """Register a health check"""
        self.checks[name] = HealthCheck(
            name=name,
            check_func=check_func,
            interval=interval_seconds,
            timeout=timeout_seconds,
            critical=critical
        )
        
    async def run_all_checks(self) -> HealthReport:
        """Run all registered health checks"""
        results = {}
        for name, check in self.checks.items():
            try:
                result = await asyncio.wait_for(
                    check.check_func(),
                    timeout=check.timeout
                )
                results[name] = HealthCheckResult(
                    status="healthy",
                    latency_ms=result.latency_ms,
                    details=result.details
                )
            except asyncio.TimeoutError:
                results[name] = HealthCheckResult(
                    status="timeout",
                    details={"error": "Check timed out"}
                )
            except Exception as e:
                results[name] = HealthCheckResult(
                    status="unhealthy",
                    details={"error": str(e)}
                )
                
        return HealthReport(
            timestamp=datetime.now(),
            overall_status=self._compute_overall_status(results),
            checks=results
        )
    
    # Standard health checks
    async def check_agent_responsive(self, agent_id: str):
        """Check if an agent responds to ping"""
        pass
        
    async def check_tool_available(self, tool_name: str):
        """Check if a tool is available and working"""
        pass
        
    async def check_disk_space(self, min_gb: float = 1.0):
        """Check available disk space"""
        pass
        
    async def check_memory_available(self, min_mb: float = 512):
        """Check available memory"""
        pass
        
    async def check_message_queue_healthy(self):
        """Check message queue is processing"""
        pass
```

---

## 6. Runbooks for Common Issues

### 6.1 Agent Not Responding

```markdown
## Runbook: Agent Not Responding

**Symptoms:**
- Health checks failing for agent
- Messages timing out
- Task queue building up

**Diagnosis Steps:**
1. Check agent logs for last activity
   ```bash
   tail -100 logs/agent-{id}.log
   ```
2. Check system resources
   ```bash
   ps aux | grep agent-{id}
   top -p {pid}
   ```
3. Check message queue for stuck messages
   ```bash
   mcp-cli queue inspect --agent {id}
   ```

**Resolution Steps:**

**Level 1 - Soft Restart:**
1. Send graceful shutdown signal
2. Wait 30 seconds
3. Restart agent process
4. Verify health checks pass

**Level 2 - Hard Restart:**
1. Force kill agent process
2. Clear agent's local state cache
3. Restart from last checkpoint
4. Re-assign pending tasks

**Level 3 - Full Recovery:**
1. Take diagnostic snapshot
2. Kill all related processes
3. Restore from known-good checkpoint
4. Restart with minimal task load
5. Gradually increase load

**Post-Incident:**
- Review logs for root cause
- Update runbook if new failure mode
- Consider adding new health check
```

### 6.2 Task Stuck in Loop

```markdown
## Runbook: Task Stuck in Loop

**Symptoms:**
- Same task retrying repeatedly
- No progress on task queue
- High CPU with no results

**Diagnosis Steps:**
1. Check task retry count
   ```bash
   mcp-cli task inspect {task_id}
   ```
2. Check for circular dependencies
   ```bash
   mcp-cli task graph --task {task_id}
   ```
3. Review last N attempts
   ```bash
   mcp-cli task history {task_id} --last 10
   ```

**Resolution Steps:**

**If Retry Loop:**
1. Pause task
2. Check if error is transient
3. If permanent error, mark task as failed
4. Notify dependent tasks

**If Circular Dependency:**
1. Pause all tasks in cycle
2. Identify the cycle
3. Break cycle by marking one task complete/failed
4. Resume remaining tasks

**If Infinite Generation:**
1. Check termination conditions
2. Add hard stop condition
3. Cancel and re-queue with fixed parameters

**Prevention:**
- Set max_retries on all tasks
- Add cycle detection to scheduler
- Implement timeout at task level
```

### 6.3 Checkpoint Corruption

```markdown
## Runbook: Checkpoint Corruption

**Symptoms:**
- Checkpoint restore fails
- Hash verification errors
- Inconsistent state after restore

**Diagnosis Steps:**
1. Verify checkpoint integrity
   ```bash
   mcp-cli checkpoint verify {checkpoint_id}
   ```
2. List available checkpoints
   ```bash
   mcp-cli checkpoint list --healthy-only
   ```
3. Check disk health
   ```bash
   df -h && dmesg | tail -50
   ```

**Resolution Steps:**

**If Single Checkpoint:**
1. Find previous valid checkpoint
2. Restore from that instead
3. Re-execute operations since that checkpoint

**If Multiple Checkpoints:**
1. STOP creating new checkpoints
2. Investigate storage health
3. Run fsck if needed
4. Restore from off-site backup if available

**If No Valid Checkpoint:**
1. Enter SAFE_MODE
2. Dump current state
3. Attempt manual state reconstruction
4. REQUIRE human intervention

**Prevention:**
- Verify checkpoints after creation
- Store checkpoints on multiple media
- Test restore process regularly
```

### 6.4 Rate Limit Exceeded

```markdown
## Runbook: Rate Limit Exceeded

**Symptoms:**
- 429 responses from external APIs
- Tool calls failing with rate limit errors
- Backlog of pending requests

**Immediate Actions:**
1. Pause all affected agents
2. Check current rate limit status
   ```bash
   mcp-cli ratelimit status --all
   ```

**Resolution:**

**If Temporary Spike:**
1. Wait for rate limit reset
2. Resume with reduced concurrency
3. Gradually increase back to normal

**If Sustained Overuse:**
1. Reduce batch sizes
2. Increase delays between calls
3. Implement request queuing
4. Consider upgrading API tier

**Configuration Adjustment:**
```yaml
rate_limits:
  github_api:
    requests_per_minute: 20  # reduced from 30
    burst_size: 5
    backoff_strategy: exponential
  openai_api:
    requests_per_minute: 50
    tokens_per_minute: 40000
```
```

---

## 7. Implementation Checklist

### Phase 1: Core Protocol (Week 1-2)

- [ ] **Message Schema Implementation**
  - [ ] Define protobuf/JSON schema for all message types
  - [ ] Implement message validation
  - [ ] Add serialization/deserialization
  - [ ] Write unit tests for all message types

- [ ] **Transport Layer**
  - [ ] Implement message queue (Redis/RabbitMQ/in-memory)
  - [ ] Add message persistence
  - [ ] Implement message acknowledgment
  - [ ] Add dead letter queue

- [ ] **Agent Lifecycle**
  - [ ] Implement agent state machine
  - [ ] Add graceful startup/shutdown
  - [ ] Implement heartbeat mechanism
  - [ ] Add capability registration

### Phase 2: Debugging Infrastructure (Week 2-3)

- [ ] **Logging System**
  - [ ] Implement structured logging
  - [ ] Add log aggregation
  - [ ] Implement log search/filter
  - [ ] Add log retention policy

- [ ] **Tracing**
  - [ ] Implement distributed tracing
  - [ ] Add trace ID propagation
  - [ ] Implement trace visualization
  - [ ] Add trace sampling

- [ ] **Checkpoints**
  - [ ] Implement checkpoint creation
  - [ ] Add checkpoint verification
  - [ ] Implement checkpoint restore
  - [ ] Add checkpoint diff

### Phase 3: Self-Healing (Week 3-4)

- [ ] **Error Handling**
  - [ ] Implement error categorization
  - [ ] Add retry logic with backoff
  - [ ] Implement circuit breakers
  - [ ] Add fallback mechanisms

- [ ] **Recovery Actions**
  - [ ] Implement automatic restart
  - [ ] Add state reconstruction
  - [ ] Implement rollback
  - [ ] Add graceful degradation

- [ ] **Health Monitoring**
  - [ ] Implement health checks
  - [ ] Add health aggregation
  - [ ] Implement health-based routing
  - [ ] Add predictive health analysis

### Phase 4: Autonomous Operation (Week 4-5)

- [ ] **Autonomous Mode**
  - [ ] Implement mode detection
  - [ ] Add operation restrictions
  - [ ] Implement resource limits
  - [ ] Add safety constraints

- [ ] **Task Management**
  - [ ] Implement autonomous task queue
  - [ ] Add task prioritization
  - [ ] Implement task scheduling
  - [ ] Add task timeout handling

- [ ] **Reporting**
  - [ ] Implement activity logs
  - [ ] Add summary reports
  - [ ] Implement change tracking
  - [ ] Add notification system

### Phase 5: Testing & Validation (Week 5-6)

- [ ] **Unit Tests**
  - [ ] Test all message types
  - [ ] Test all state transitions
  - [ ] Test all recovery actions
  - [ ] Test all health checks

- [ ] **Integration Tests**
  - [ ] Test multi-agent scenarios
  - [ ] Test failure scenarios
  - [ ] Test recovery scenarios
  - [ ] Test autonomous mode

- [ ] **Chaos Engineering**
  - [ ] Implement failure injection
  - [ ] Test network partitions
  - [ ] Test resource exhaustion
  - [ ] Test cascading failures

---

## Appendix A: Quick Reference

### Status Codes

| Code | Category | Meaning |
|------|----------|---------|
| 1xx | Info | Informational responses |
| 2xx | Success | Operation successful |
| 3xx | Redirect | Action requires different path |
| 4xx | Client Error | Invalid request |
| 5xx | Server Error | Internal error |
| 6xx | External Error | External service failure |
| 7xx | Resource Error | Resource limits exceeded |
| 8xx | Safety Error | Safety violation detected |
| 9xx | System Error | System-level failure |

### Common Error Codes

| Code | Name | Recovery |
|------|------|----------|
| 4001 | INVALID_MESSAGE | Fix message format |
| 4002 | UNKNOWN_INTENT | Check intent spelling |
| 4003 | MISSING_FIELD | Add required field |
| 5001 | INTERNAL_ERROR | Retry with backoff |
| 5002 | AGENT_OVERLOADED | Reduce load |
| 6001 | EXTERNAL_TIMEOUT | Retry later |
| 6002 | RATE_LIMITED | Wait and retry |
| 7001 | MEMORY_EXCEEDED | Free resources |
| 7002 | DISK_FULL | Cleanup files |
| 8001 | SAFETY_VIOLATION | Require human approval |
| 8002 | FORBIDDEN_ACTION | Check permissions |
| 9001 | UNRECOVERABLE | Human intervention required |

### CLI Quick Commands

```bash
# Agent management
mcp-cli agent list
mcp-cli agent status {id}
mcp-cli agent restart {id}
mcp-cli agent pause {id}

# Task management
mcp-cli task list [--status pending|running|complete|failed]
mcp-cli task inspect {id}
mcp-cli task cancel {id}
mcp-cli task retry {id}

# Debugging
mcp-cli debug dump-state
mcp-cli debug trace {message_id}
mcp-cli debug set-level {DEBUG|INFO|WARN|ERROR}

# Checkpoints
mcp-cli checkpoint create [--name "description"]
mcp-cli checkpoint list
mcp-cli checkpoint restore {id}
mcp-cli checkpoint diff {id1} {id2}

# Health
mcp-cli health check
mcp-cli health report
mcp-cli health history --last 24h

# Autonomous mode
mcp-cli autonomous status
mcp-cli autonomous enable
mcp-cli autonomous disable
mcp-cli autonomous config show
```

---

## Appendix B: Configuration Templates

### Full Configuration Example

```yaml
# mcp_config.yaml - Full configuration for MCP system

system:
  name: "penta-core-mcp"
  version: "1.0.0"
  environment: "production"  # development, staging, production

agents:
  planner:
    instances: 1
    memory_limit_mb: 1024
    cpu_limit_percent: 25
    tools:
      - task_create
      - task_assign
      - task_query
      - agent_query
      
  researcher:
    instances: 2
    memory_limit_mb: 2048
    cpu_limit_percent: 50
    tools:
      - web_search
      - file_read
      - knowledge_query
      - summarize
      
  builder:
    instances: 2
    memory_limit_mb: 4096
    cpu_limit_percent: 75
    tools:
      - file_read
      - file_write
      - code_execute
      - test_run
      
  tester:
    instances: 1
    memory_limit_mb: 2048
    cpu_limit_percent: 50
    tools:
      - test_run
      - coverage_report
      - benchmark
      
  auditor:
    instances: 1
    memory_limit_mb: 1024
    cpu_limit_percent: 25
    tools:
      - log_query
      - security_scan
      - cost_report

message_queue:
  type: "redis"  # redis, rabbitmq, memory
  host: "localhost"
  port: 6379
  db: 0
  max_retries: 3
  retry_delay_ms: 1000

checkpoints:
  enabled: true
  storage: "filesystem"  # filesystem, s3, gcs
  path: "./checkpoints"
  interval_seconds: 300
  max_count: 100
  verify_on_create: true

logging:
  level: "INFO"  # TRACE, DEBUG, INFO, WARN, ERROR
  format: "json"
  outputs:
    - type: "file"
      path: "./logs/mcp.log"
      rotation: "daily"
      retention_days: 30
    - type: "stdout"
      format: "pretty"

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval_seconds: 30
  alert_webhook: "https://hooks.example.com/alerts"

autonomous:
  enabled: true
  activation_delay_minutes: 5
  max_runtime_hours: 8
  daily_budget_dollars: 10.00
  require_approval_for:
    - external_api_calls
    - file_deletions
    - deployments

safety:
  max_consecutive_failures: 5
  circuit_breaker_threshold: 10
  emergency_stop_enabled: true
  require_human_for:
    - production_changes
    - security_modifications
    - budget_increases
```

---

*Document Version: 1.0.0*
*Last Updated: 2024*
*Maintainer: Penta-Core Team*
