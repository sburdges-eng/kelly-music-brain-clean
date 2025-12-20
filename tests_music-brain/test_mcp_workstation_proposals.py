"""
Tests for MCP Workstation Proposal Management.

Tests the ProposalManager class and proposal-related utilities.
"""

import pytest
from unittest.mock import patch, MagicMock

from mcp_workstation.models import (
    Proposal,
    ProposalStatus,
    ProposalCategory,
    AIAgent,
)
from mcp_workstation.proposals import (
    ProposalManager,
    ProposalVote,
    get_proposal_template,
    format_proposal,
    format_proposal_list,
)


@pytest.fixture
def manager():
    """Create a fresh ProposalManager."""
    return ProposalManager()


@pytest.fixture
def manager_with_proposals():
    """Create a ProposalManager with some proposals."""
    manager = ProposalManager()

    # Add proposals from different agents
    manager.submit_proposal(
        agent=AIAgent.CLAUDE,
        title="Architecture Improvement",
        description="Improve the system architecture",
        category=ProposalCategory.ARCHITECTURE,
        priority=8,
    )
    manager.submit_proposal(
        agent=AIAgent.CHATGPT,
        title="Performance Optimization",
        description="Optimize performance",
        category=ProposalCategory.PERFORMANCE,
        priority=7,
    )
    manager.submit_proposal(
        agent=AIAgent.GEMINI,
        title="New Feature",
        description="Add new feature",
        category=ProposalCategory.FEATURE_NEW,
        priority=6,
    )

    return manager


class TestProposalVote:
    """Tests for ProposalVote dataclass."""

    def test_vote_creation(self):
        """Test creating a vote."""
        vote = ProposalVote(
            agent=AIAgent.CLAUDE,
            proposal_id="p123",
            vote=1,
        )
        assert vote.agent == AIAgent.CLAUDE
        assert vote.proposal_id == "p123"
        assert vote.vote == 1
        assert vote.timestamp != ""

    def test_vote_with_comment(self):
        """Test vote with comment."""
        vote = ProposalVote(
            agent=AIAgent.CHATGPT,
            proposal_id="p123",
            vote=-1,
            comment="Disagree with approach",
        )
        assert vote.comment == "Disagree with approach"


class TestProposalManagerInit:
    """Tests for ProposalManager initialization."""

    def test_empty_init(self, manager):
        """Test empty initialization."""
        assert len(manager.proposals) == 0
        assert len(manager.votes) == 0

    def test_max_proposals_per_agent(self, manager):
        """Test max proposals constant."""
        assert manager.MAX_PROPOSALS_PER_AGENT == 3


class TestProposalManagerSubmit:
    """Tests for submitting proposals."""

    def test_submit_proposal(self, manager):
        """Test submitting a proposal."""
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Test Proposal",
            description="Description",
            category=ProposalCategory.FEATURE_NEW,
        )
        assert proposal is not None
        assert proposal.title == "Test Proposal"
        assert proposal.status == ProposalStatus.SUBMITTED
        assert proposal.id in manager.proposals

    def test_submit_with_all_fields(self, manager):
        """Test submitting with all fields."""
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Full Proposal",
            description="Complete description",
            category=ProposalCategory.ARCHITECTURE,
            priority=9,
            estimated_effort="high",
            phase_target=2,
            implementation_notes="Notes here",
        )
        assert proposal.priority == 9
        assert proposal.estimated_effort == "high"
        assert proposal.phase_target == 2
        assert proposal.implementation_notes == "Notes here"

    def test_submit_limit_per_agent(self, manager):
        """Test that agents are limited to MAX_PROPOSALS_PER_AGENT."""
        # Submit max proposals
        for i in range(manager.MAX_PROPOSALS_PER_AGENT):
            proposal = manager.submit_proposal(
                agent=AIAgent.CLAUDE,
                title=f"Proposal {i}",
                description="Desc",
                category=ProposalCategory.FEATURE_NEW,
            )
            assert proposal is not None

        # Try to submit one more
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Over Limit",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        assert proposal is None

    def test_submit_different_agents(self, manager):
        """Test that different agents have separate limits."""
        # Claude submits max
        for i in range(manager.MAX_PROPOSALS_PER_AGENT):
            manager.submit_proposal(
                agent=AIAgent.CLAUDE,
                title=f"Claude {i}",
                description="Desc",
                category=ProposalCategory.FEATURE_NEW,
            )

        # ChatGPT should still be able to submit
        proposal = manager.submit_proposal(
            agent=AIAgent.CHATGPT,
            title="ChatGPT Proposal",
            description="Desc",
            category=ProposalCategory.FEATURE_NEW,
        )
        assert proposal is not None


class TestProposalManagerVoting:
    """Tests for voting on proposals."""

    def test_vote_on_proposal(self, manager_with_proposals):
        """Test voting on a proposal."""
        proposal_id = list(manager_with_proposals.proposals.keys())[0]
        proposal = manager_with_proposals.proposals[proposal_id]

        # Vote from different agent
        if proposal.agent != AIAgent.CHATGPT:
            result = manager_with_proposals.vote_on_proposal(
                agent=AIAgent.CHATGPT,
                proposal_id=proposal_id,
                vote=1,
            )
            assert result is True
            assert proposal.votes["chatgpt"] == 1

    def test_vote_with_comment(self, manager_with_proposals):
        """Test voting with a comment."""
        proposal_id = list(manager_with_proposals.proposals.keys())[0]
        proposal = manager_with_proposals.proposals[proposal_id]

        if proposal.agent != AIAgent.GEMINI:
            result = manager_with_proposals.vote_on_proposal(
                agent=AIAgent.GEMINI,
                proposal_id=proposal_id,
                vote=1,
                comment="Great idea!",
            )
            assert result is True

        # Check vote was recorded
        votes = [v for v in manager_with_proposals.votes if v.proposal_id == proposal_id]
        if votes:
            assert any(v.comment == "Great idea!" for v in votes)

    def test_cannot_vote_on_own_proposal(self, manager_with_proposals):
        """Test that agents cannot vote on their own proposals."""
        # Find Claude's proposal
        claude_proposal = next(
            p for p in manager_with_proposals.proposals.values()
            if p.agent == AIAgent.CLAUDE
        )

        result = manager_with_proposals.vote_on_proposal(
            agent=AIAgent.CLAUDE,
            proposal_id=claude_proposal.id,
            vote=1,
        )
        assert result is False

    def test_vote_nonexistent_proposal(self, manager):
        """Test voting on non-existent proposal."""
        result = manager.vote_on_proposal(
            agent=AIAgent.CLAUDE,
            proposal_id="nonexistent",
            vote=1,
        )
        assert result is False

    def test_consensus_approval(self, manager):
        """Test that proposal is approved with consensus."""
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Consensus Test",
            description="Test",
            category=ProposalCategory.FEATURE_NEW,
        )

        # All other agents approve
        for agent in AIAgent:
            if agent != AIAgent.CLAUDE:
                manager.vote_on_proposal(agent, proposal.id, 1)

        assert proposal.status == ProposalStatus.APPROVED

    def test_consensus_rejection(self, manager):
        """Test that proposal is rejected with consensus."""
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Rejection Test",
            description="Test",
            category=ProposalCategory.FEATURE_NEW,
        )

        # All other agents reject
        for agent in AIAgent:
            if agent != AIAgent.CLAUDE:
                manager.vote_on_proposal(agent, proposal.id, -1)

        assert proposal.status == ProposalStatus.REJECTED

    def test_mixed_votes_under_review(self, manager):
        """Test that mixed votes put proposal under review."""
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Mixed Test",
            description="Test",
            category=ProposalCategory.FEATURE_NEW,
        )

        # Mixed votes
        manager.vote_on_proposal(AIAgent.CHATGPT, proposal.id, 1)
        manager.vote_on_proposal(AIAgent.GEMINI, proposal.id, -1)
        manager.vote_on_proposal(AIAgent.GITHUB_COPILOT, proposal.id, 0)

        assert proposal.status == ProposalStatus.UNDER_REVIEW


class TestProposalManagerUpdateStatus:
    """Tests for updating proposal status."""

    def test_update_status(self, manager_with_proposals):
        """Test updating a proposal's status."""
        proposal_id = list(manager_with_proposals.proposals.keys())[0]

        result = manager_with_proposals.update_status(
            proposal_id,
            ProposalStatus.IMPLEMENTED,
        )
        assert result is True
        assert manager_with_proposals.proposals[proposal_id].status == ProposalStatus.IMPLEMENTED

    def test_update_status_with_notes(self, manager_with_proposals):
        """Test updating status with notes."""
        proposal_id = list(manager_with_proposals.proposals.keys())[0]

        manager_with_proposals.update_status(
            proposal_id,
            ProposalStatus.DEFERRED,
            notes="Postponed to next sprint",
        )
        proposal = manager_with_proposals.proposals[proposal_id]
        assert "Postponed" in proposal.implementation_notes

    def test_update_nonexistent(self, manager):
        """Test updating non-existent proposal."""
        result = manager.update_status("nonexistent", ProposalStatus.APPROVED)
        assert result is False


class TestProposalManagerQueries:
    """Tests for query methods."""

    def test_get_proposal(self, manager_with_proposals):
        """Test getting a proposal by ID."""
        proposal_id = list(manager_with_proposals.proposals.keys())[0]
        proposal = manager_with_proposals.get_proposal(proposal_id)
        assert proposal is not None

    def test_get_nonexistent_proposal(self, manager):
        """Test getting non-existent proposal."""
        proposal = manager.get_proposal("nonexistent")
        assert proposal is None

    def test_get_all_proposals(self, manager_with_proposals):
        """Test getting all proposals."""
        proposals = manager_with_proposals.get_all_proposals()
        assert len(proposals) == 3

    def test_get_proposals_by_agent(self, manager_with_proposals):
        """Test getting proposals by agent."""
        proposals = manager_with_proposals.get_proposals_by_agent(AIAgent.CLAUDE)
        assert len(proposals) == 1
        assert proposals[0].agent == AIAgent.CLAUDE

    def test_get_proposals_by_status(self, manager_with_proposals):
        """Test getting proposals by status."""
        proposals = manager_with_proposals.get_proposals_by_status(ProposalStatus.SUBMITTED)
        assert len(proposals) == 3  # All are submitted

    def test_get_proposals_by_category(self, manager_with_proposals):
        """Test getting proposals by category."""
        proposals = manager_with_proposals.get_proposals_by_category(
            ProposalCategory.ARCHITECTURE
        )
        assert len(proposals) == 1

    def test_get_proposals_by_phase(self, manager):
        """Test getting proposals by phase target."""
        manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Phase 1",
            description="",
            category=ProposalCategory.FEATURE_NEW,
            phase_target=1,
        )
        manager.submit_proposal(
            agent=AIAgent.CHATGPT,
            title="Phase 2",
            description="",
            category=ProposalCategory.FEATURE_NEW,
            phase_target=2,
        )

        phase1 = manager.get_proposals_by_phase(1)
        phase2 = manager.get_proposals_by_phase(2)

        assert len(phase1) == 1
        assert len(phase2) == 1

    def test_get_approved_proposals(self, manager):
        """Test getting approved proposals sorted by priority."""
        p1 = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Low Priority",
            description="",
            category=ProposalCategory.FEATURE_NEW,
            priority=3,
        )
        p2 = manager.submit_proposal(
            agent=AIAgent.CHATGPT,
            title="High Priority",
            description="",
            category=ProposalCategory.FEATURE_NEW,
            priority=9,
        )

        # Approve both
        manager.update_status(p1.id, ProposalStatus.APPROVED)
        manager.update_status(p2.id, ProposalStatus.APPROVED)

        approved = manager.get_approved_proposals()
        assert len(approved) == 2
        assert approved[0].priority > approved[1].priority

    def test_get_pending_votes(self, manager):
        """Test getting proposals an agent hasn't voted on."""
        # Create proposal from Claude
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Needs Votes",
            description="",
            category=ProposalCategory.FEATURE_NEW,
        )

        # ChatGPT should have pending vote
        pending = manager.get_pending_votes(AIAgent.CHATGPT)
        assert len(pending) == 1

        # Vote
        manager.vote_on_proposal(AIAgent.CHATGPT, proposal.id, 1)

        # No more pending
        pending = manager.get_pending_votes(AIAgent.CHATGPT)
        assert len(pending) == 0


class TestProposalManagerAnalysis:
    """Tests for analysis methods."""

    def test_get_proposal_summary(self, manager_with_proposals):
        """Test getting proposal summary."""
        summary = manager_with_proposals.get_proposal_summary()

        assert summary["total"] == 3
        assert "by_status" in summary
        assert "by_agent" in summary
        assert "by_category" in summary
        assert "by_phase" in summary

    def test_get_implementation_queue(self, manager):
        """Test getting prioritized implementation queue."""
        p1 = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="P1",
            description="",
            category=ProposalCategory.FEATURE_NEW,
            priority=5,
        )
        p2 = manager.submit_proposal(
            agent=AIAgent.CHATGPT,
            title="P2",
            description="",
            category=ProposalCategory.FEATURE_NEW,
            priority=8,
        )

        manager.update_status(p1.id, ProposalStatus.APPROVED)
        manager.update_status(p2.id, ProposalStatus.APPROVED)

        queue = manager.get_implementation_queue()
        assert len(queue) == 2
        assert queue[0].priority >= queue[1].priority

    def test_get_agent_proposal_slots(self, manager_with_proposals):
        """Test getting remaining proposal slots."""
        slots = manager_with_proposals.get_agent_proposal_slots()

        # Each agent has 1 proposal, so 2 slots remaining each
        assert slots[AIAgent.CLAUDE] == 2
        assert slots[AIAgent.CHATGPT] == 2
        assert slots[AIAgent.GEMINI] == 2
        assert slots[AIAgent.GITHUB_COPILOT] == 3  # None submitted


class TestProposalManagerSerialization:
    """Tests for serialization."""

    def test_to_dict(self, manager_with_proposals):
        """Test serializing manager to dictionary."""
        data = manager_with_proposals.to_dict()

        assert "proposals" in data
        assert "votes" in data
        assert len(data["proposals"]) == 3

    def test_from_dict(self):
        """Test deserializing manager from dictionary."""
        data = {
            "proposals": {
                "p1": {
                    "id": "p1",
                    "agent": "claude",
                    "title": "Test",
                    "description": "Desc",
                    "category": "feature_new",
                    "status": "approved",
                    "priority": 7,
                    "votes": {"chatgpt": 1},
                }
            },
            "votes": [
                {
                    "agent": "chatgpt",
                    "proposal_id": "p1",
                    "vote": 1,
                    "comment": "Good",
                    "timestamp": "2024-01-01T00:00:00",
                }
            ],
        }
        manager = ProposalManager.from_dict(data)

        assert len(manager.proposals) == 1
        assert len(manager.votes) == 1
        assert manager.proposals["p1"].status == ProposalStatus.APPROVED

    def test_roundtrip(self, manager_with_proposals):
        """Test serialization roundtrip."""
        # Add some votes
        proposal_id = list(manager_with_proposals.proposals.keys())[0]
        proposal = manager_with_proposals.proposals[proposal_id]
        if proposal.agent != AIAgent.GITHUB_COPILOT:
            manager_with_proposals.vote_on_proposal(
                AIAgent.GITHUB_COPILOT, proposal_id, 1
            )

        data = manager_with_proposals.to_dict()
        restored = ProposalManager.from_dict(data)

        assert len(restored.proposals) == len(manager_with_proposals.proposals)
        assert len(restored.votes) == len(manager_with_proposals.votes)


class TestProposalTemplates:
    """Tests for proposal templates."""

    def test_architecture_template(self):
        """Test architecture proposal template."""
        template = get_proposal_template(ProposalCategory.ARCHITECTURE)

        assert "[Architecture]" in template["title"]
        assert "Problem" in template["description_template"]
        assert template["default_effort"] == "high"

    def test_performance_template(self):
        """Test performance proposal template."""
        template = get_proposal_template(ProposalCategory.PERFORMANCE)

        assert "[Performance]" in template["title"]
        assert "Bottleneck" in template["description_template"]

    def test_cpp_port_template(self):
        """Test C++ port proposal template."""
        template = get_proposal_template(ProposalCategory.CPP_PORT)

        assert "[C++ Port]" in template["title"]
        assert "Python Bridge" in template["description_template"]
        assert template["default_effort"] == "very_high"

    def test_dsp_template(self):
        """Test DSP algorithm template."""
        template = get_proposal_template(ProposalCategory.DSP_ALGORITHM)

        assert "[DSP]" in template["title"]
        assert "Sample rate" in template["description_template"]

    def test_unknown_category_template(self):
        """Test that unknown categories get default template."""
        template = get_proposal_template(ProposalCategory.DOCUMENTATION)

        assert "documentation" in template["title"]
        assert template["default_effort"] == "medium"


class TestFormatProposal:
    """Tests for format_proposal function."""

    def test_format_includes_title(self, manager_with_proposals):
        """Test that format includes title."""
        proposal = list(manager_with_proposals.proposals.values())[0]
        output = format_proposal(proposal)

        assert proposal.title in output

    def test_format_includes_details(self, manager_with_proposals):
        """Test that format includes all details."""
        proposal = list(manager_with_proposals.proposals.values())[0]
        output = format_proposal(proposal)

        assert proposal.id in output
        assert proposal.agent.display_name in output
        assert proposal.category.value in output
        assert proposal.status.value in output

    def test_format_includes_votes(self, manager):
        """Test that format includes votes when present."""
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Voted",
            description="Has votes",
            category=ProposalCategory.FEATURE_NEW,
        )
        manager.vote_on_proposal(AIAgent.CHATGPT, proposal.id, 1)

        output = format_proposal(proposal, include_votes=True)
        assert "VOTES" in output or "chatgpt" in output

    def test_format_without_votes(self, manager):
        """Test format without votes."""
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="No Votes",
            description="Test",
            category=ProposalCategory.FEATURE_NEW,
        )
        manager.vote_on_proposal(AIAgent.CHATGPT, proposal.id, 1)

        output = format_proposal(proposal, include_votes=False)
        # Votes section might not appear


class TestFormatProposalList:
    """Tests for format_proposal_list function."""

    def test_format_list_empty(self):
        """Test formatting empty list."""
        output = format_proposal_list([])
        assert "No proposals" in output

    def test_format_list_with_proposals(self, manager_with_proposals):
        """Test formatting list with proposals."""
        proposals = manager_with_proposals.get_all_proposals()
        output = format_proposal_list(proposals)

        assert "ID" in output
        assert "Status" in output
        assert "Agent" in output


class TestProposalManagerIntegration:
    """Integration tests for ProposalManager."""

    def test_full_proposal_lifecycle(self, manager):
        """Test a complete proposal lifecycle."""
        # Submit
        proposal = manager.submit_proposal(
            agent=AIAgent.CLAUDE,
            title="Lifecycle Test",
            description="Testing full lifecycle",
            category=ProposalCategory.FEATURE_NEW,
            priority=7,
        )
        assert proposal.status == ProposalStatus.SUBMITTED

        # Voting
        manager.vote_on_proposal(AIAgent.CHATGPT, proposal.id, 1)
        manager.vote_on_proposal(AIAgent.GEMINI, proposal.id, 1)
        manager.vote_on_proposal(AIAgent.GITHUB_COPILOT, proposal.id, 1)

        # Should be approved
        assert proposal.status == ProposalStatus.APPROVED

        # Implementation
        manager.update_status(
            proposal.id,
            ProposalStatus.IMPLEMENTED,
            notes="Completed in sprint 5",
        )
        assert proposal.status == ProposalStatus.IMPLEMENTED

    def test_multi_agent_proposal_submission(self, manager):
        """Test multiple agents submitting proposals."""
        agents = [AIAgent.CLAUDE, AIAgent.CHATGPT, AIAgent.GEMINI, AIAgent.GITHUB_COPILOT]

        for agent in agents:
            manager.submit_proposal(
                agent=agent,
                title=f"{agent.value} proposal",
                description="Agent's proposal",
                category=ProposalCategory.FEATURE_NEW,
            )

        assert len(manager.proposals) == 4
        summary = manager.get_proposal_summary()
        assert summary["by_agent"]["claude"] == 1
        assert summary["by_agent"]["chatgpt"] == 1
        assert summary["by_agent"]["gemini"] == 1
        assert summary["by_agent"]["github_copilot"] == 1

    def test_cross_voting_scenario(self, manager):
        """Test cross-voting between agents."""
        # Each agent submits a proposal
        proposals = {}
        for agent in [AIAgent.CLAUDE, AIAgent.CHATGPT]:
            p = manager.submit_proposal(
                agent=agent,
                title=f"{agent.value}'s idea",
                description="Great idea",
                category=ProposalCategory.FEATURE_NEW,
            )
            proposals[agent] = p

        # Agents vote on each other's proposals
        manager.vote_on_proposal(AIAgent.CHATGPT, proposals[AIAgent.CLAUDE].id, 1)
        manager.vote_on_proposal(AIAgent.CLAUDE, proposals[AIAgent.CHATGPT].id, 1)

        # Check votes recorded
        assert proposals[AIAgent.CLAUDE].votes.get("chatgpt") == 1
        assert proposals[AIAgent.CHATGPT].votes.get("claude") == 1
