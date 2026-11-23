"""Neutral data model for candidate-job matching.

The pipeline never reads or copies a source system's native format. A source adapter (see `adapter.py`)
maps whatever a system or data source exports into these records. Field names here are this
project's own vocabulary that can be customized or redefined in other ways; a real deployment maps 
its source's identifiers onto them, and those identifiers will differ from system to system.

Four record types:

    Candidate     a person, with one or more captured resume versions
    Job           an opening at a hiring organization, with its description and internal notes
    Application   one candidate submitted to one job, with the stage history that followed
                  (stage names come from a configurable funnel, see funnel.py)
    Snapshot      the manifest tying one export together (id, reference date)

Privacy split: `Candidate.identity` and `Job.related_people` exist only so the de-identification
pipeline knows which spans to mask. They are dropped before anything reaches a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"                    # stage finished and the process moved on
    DROPPED = "dropped"                  # process ended at this stage (see Application.end_reason)


class EndReason:
    """Default vocabulary for why a process ended before a hire.

    Ending is not evidence of a poor fit by itself. Which reasons are excluded from labels, or
    imply that the employer reviewed the candidate, is decided by the funnel (see funnel.py).
    """

    EMPLOYER_DECLINED = "employer_declined"
    INTERNAL_SCREEN_OUT = "internal_screen_out"
    CANDIDATE_WITHDREW = "candidate_withdrew"
    POSITION_CLOSED = "position_closed"
    AUTO_EXPIRED = "auto_expired"
    DUPLICATE = "duplicate"
    ENTRY_ERROR = "entry_error"
    OTHER = "other"


@dataclass
class Section:
    kind: str                            # experience | education | skills | projects | summary | other
    text: str


@dataclass
class ResumeVersion:
    resume_id: str
    captured_at: str | None              # ISO date the file entered the system, if known
    text: str
    sections: list[Section] = field(default_factory=list)
    language: str | None = None          # e.g. "en", "zh", "mixed"


@dataclass
class Identity:
    """De-identification input only. Never written to model inputs."""

    full_name: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    other_names: list[str] = field(default_factory=list)   # nicknames, native-script names
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    candidate_id: str
    resumes: list[ResumeVersion] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    identity: Identity = field(default_factory=Identity)


@dataclass
class Compensation:
    period: str | None = None            # yearly | monthly | hourly
    minimum: float | None = None
    maximum: float | None = None
    currency: str | None = None


@dataclass
class JobNote:
    note_id: str
    text: str
    written_at: str | None = None        # ISO datetime; None means the note cannot be time-checked


@dataclass
class Job:
    job_id: str
    org_id: str                          # hiring organization
    org_name: str
    title: str
    employment_type: str | None = None   # full_time | contract | intern | ...
    locations: list[str] = field(default_factory=list)
    compensation: Compensation = field(default_factory=Compensation)
    summary: str = ""
    responsibilities: str = ""
    requirements: str = ""
    required_skills: list[str] = field(default_factory=list)
    opened_at: str | None = None         # ISO date; used for the "open at as-of" job pool
    closed_at: str | None = None
    status: str | None = None            # outcome-like; never shown to models
    notes: list[JobNote] = field(default_factory=list)
    related_people: list[str] = field(default_factory=list)  # de-identification input only


@dataclass
class StageEvent:
    stage: str                           # a stage name defined by the snapshot's funnel
    status: StageStatus
    at: str | None = None                # ISO datetime of the last change


@dataclass
class Application:
    application_id: str
    candidate_id: str
    job_id: str
    submitted_at: str                    # ISO datetime; the as-of anchor
    stages: list[StageEvent] = field(default_factory=list)
    end_reason: str | None = None        # see EndReason for the default vocabulary
    end_note: str = ""                   # free text; outcome-like, never shown to models
    resume_id: str | None = None         # resume version sent with this submission, if recorded
    updated_at: str | None = None


@dataclass
class Snapshot:
    snapshot_id: str
    reference_date: str                  # ISO date the export was taken; drives label maturity
    source: str = "synthetic"            # free-text tag naming the adapter that produced it
    funnel: str = "staffing"             # name of the FunnelSpec that describes the stages
    schema_version: str = "1.0"
