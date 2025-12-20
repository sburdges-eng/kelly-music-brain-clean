from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import (
    AIAgent,
    Proposal,
    ProposalCategory,
    ProposalStatus,
    ProposalVote,
)


def _template(title: str, description: str, effort: str) -> Dict[str, str]:
    return {
        "title": title,
        "description_template": description,
        "default_effort": effort,
    }


TEMPLATES = {
    ProposalCategory.ARCHITECTURE: _template(
        "[Architecture] Improve subsystem",
        "Problem: \nSolution: \nImpact:",
        "high",
    ),
    ProposalCategory.PERFORMANCE: _template(
        "[Performance] Optimize hot path",
        "Bottleneck: \nProposal: \nRisks:",
        "medium",
    ),
    ProposalCategory.CPP_PORT: _template(
        "[C++ Port] Move component to C++",
        "Target: \nInterop: Python Bridge considerations\nTesting:",
        "very_high",
    ),
    ProposalCategory.DSP_ALGORITHM: _template(
        "[DSP] Algorithm proposal",
        "Sample rate: \nLatency: \nQuality:",
        "high",
    ),
}


def get_proposal_template(category: ProposalCategory) -> Dict[str, str]:
    return TEMPLATES.get(
        category,
        _template(
            f"[{category.value}] Proposal",
            "Problem: \nSolution: \nImpact:",
            "medium",
        ),
    )


def format_proposal(proposal: Proposal, include_votes: bool = True) -> str:
    lines = [
        f"{proposal.title} ({proposal.id})",
        f"Agent: {proposal.agent.display_name}",
        f"Category: {proposal.category.value}",
        f"Status: {proposal.status.value}",
        f"Priority: {proposal.priority}",
    ]
    if include_votes and proposal.votes:
        lines.append("VOTES:")
        for agent, vote in proposal.votes.items():
            lines.append(f" - {agent}: {vote}")
    return "\n".join(lines)


def format_proposal_list(proposals: List[Proposal]) -> str:
    if not proposals:
        return "No proposals available."
    lines = ["ID | Title | Status | Agent"]
    for p in proposals:
        lines.append(f"{p.id} | {p.title} | {p.status.value} | {p.agent.value}")
    return "\n".join(lines)


class ProposalManager:
    MAX_PROPOSALS_PER_AGENT = 3

    def __init__(self):
        self.proposals: Dict[str, Proposal] = {}
        self.votes: List[ProposalVote] = []

    def submit_proposal(
        self,
        agent: AIAgent,
        title: str,
        description: str,
        category: ProposalCategory,
        priority: int = 5,
        estimated_effort: str = "medium",
        phase_target: int = 1,
        implementation_notes: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> Optional[Proposal]:
        # enforce per-agent limit
        count = len([p for p in self.proposals.values() if p.agent == agent])
        if count >= self.MAX_PROPOSALS_PER_AGENT:
            return None
        proposal = Proposal(
            id="",
            agent=agent,
            title=title,
            description=description,
            category=category,
            status=ProposalStatus.SUBMITTED,
            priority=priority,
            estimated_effort=estimated_effort,
            phase_target=phase_target,
            implementation_notes=implementation_notes,
            dependencies=dependencies or [],
        )
        self.proposals[proposal.id] = proposal
        return proposal

    def vote_on_proposal(self, agent: AIAgent, proposal_id: str, vote: int, comment: Optional[str] = None) -> bool:
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False
        if proposal.agent == agent:
            return False
        proposal.add_vote(agent, vote)
        self.votes.append(ProposalVote(agent=agent, proposal_id=proposal_id, vote=vote, comment=comment))

        # Determine status based on other agents' votes
        other_agents = [a for a in AIAgent if a != proposal.agent]
        votes = [proposal.votes.get(a.value, 0) for a in other_agents]
        if all(v == 1 for v in votes):
            proposal.status = ProposalStatus.APPROVED
        elif all(v == -1 for v in votes):
            proposal.status = ProposalStatus.REJECTED
        elif any(v != 0 for v in votes):
            proposal.status = ProposalStatus.UNDER_REVIEW
        return True

    def update_status(self, proposal_id: str, status: ProposalStatus, notes: Optional[str] = None) -> bool:
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False
        proposal.status = status
        if notes:
            proposal.implementation_notes = (proposal.implementation_notes + "\n" + notes).strip()
        return True

    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        return self.proposals.get(proposal_id)

    def get_all_proposals(self) -> List[Proposal]:
        return list(self.proposals.values())

    def get_proposals_by_agent(self, agent: AIAgent) -> List[Proposal]:
        return [p for p in self.proposals.values() if p.agent == agent]

    def get_proposals_by_status(self, status: ProposalStatus) -> List[Proposal]:
        return [p for p in self.proposals.values() if p.status == status]

    def get_proposals_by_category(self, category: ProposalCategory) -> List[Proposal]:
        return [p for p in self.proposals.values() if p.category == category]

    def get_proposals_by_phase(self, phase_target: int) -> List[Proposal]:
        return [p for p in self.proposals.values() if p.phase_target == phase_target]

    def get_approved_proposals(self) -> List[Proposal]:
        return sorted(
            [p for p in self.proposals.values() if p.status == ProposalStatus.APPROVED],
            key=lambda p: p.priority,
            reverse=True,
        )

    def get_implementation_queue(self) -> List[Proposal]:
        return self.get_approved_proposals()

    def get_agent_proposal_slots(self) -> Dict[AIAgent, int]:
        slots: Dict[AIAgent, int] = {}
        for agent in AIAgent:
            used = len([p for p in self.proposals.values() if p.agent == agent])
            slots[agent] = self.MAX_PROPOSALS_PER_AGENT - used
        return slots

    def get_pending_votes(self, agent: AIAgent) -> List[Proposal]:
        return [
            p
            for p in self.proposals.values()
            if p.agent != agent and agent.value not in p.votes
        ]

    def to_dict(self) -> Dict[str, any]:
        return {
            "proposals": {pid: p.to_dict() for pid, p in self.proposals.items()},
            "votes": [v.to_dict() for v in self.votes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "ProposalManager":
        mgr = cls()
        mgr.proposals = {pid: Proposal.from_dict(pdata) for pid, pdata in data.get("proposals", {}).items()}
        mgr.votes = [ProposalVote.from_dict(v) for v in data.get("votes", [])]
        return mgr

    def get_proposal_summary(self) -> Dict[str, Dict[str, int]]:
        by_agent: Dict[str, int] = {a.value: 0 for a in AIAgent}
        by_status: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        by_phase: Dict[int, int] = {}
        for p in self.proposals.values():
            by_agent[p.agent.value] = by_agent.get(p.agent.value, 0) + 1
            by_status[p.status.value] = by_status.get(p.status.value, 0) + 1
            by_category[p.category.value] = by_category.get(p.category.value, 0) + 1
            by_phase[p.phase_target] = by_phase.get(p.phase_target, 0) + 1
        return {
            "total": len(self.proposals),
            "by_agent": by_agent,
            "by_status": by_status,
            "by_category": by_category,
            "by_phase": by_phase,
        }
