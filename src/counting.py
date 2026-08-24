"""Vehicle counting via virtual counting lines.

A counting line is a segment. A track is counted the first time its
trajectory crosses that segment, attributed to whichever side ("direction A"
or "direction B") it ended up on. Each track id is counted at most once per
line — per spec section 13 ("avoid counting the same tracking ID multiple
times"), even a vehicle that idles near the line and crosses back and forth
is still one vehicle, counted once.

Pure geometry + track-history bookkeeping — no video/model dependency, so
it's independently unit-testable (see tests/test_counting.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


def _side(p1: Point, p2: Point, point: Point) -> float:
    """Signed area (2D cross product) of (p2-p1) x (point-p1) — its sign says
    which side of the infinite line through p1->p2 `point` falls on."""
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1-p2 and segment p3-p4 actually cross (not just that
    the infinite lines through them would). Standard signed-area test; exact
    collinearity (a side value of precisely 0.0) is treated as "not crossing"
    rather than specially handled — real detection-center coordinates are
    floats and essentially never land exactly on the line, so this is not a
    practical concern for this use case.
    """
    d1 = _side(p3, p4, p1)
    d2 = _side(p3, p4, p2)
    d3 = _side(p1, p2, p3)
    d4 = _side(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass
class CountingLine:
    """A virtual line vehicles are counted against when crossed."""

    name: str
    p1: Point
    p2: Point
    label_a: str = "A"
    label_b: str = "B"

    def side_of(self, point: Point) -> str:
        return self.label_a if _side(self.p1, self.p2, point) > 0 else self.label_b

    def crossed_by(self, prev_point: Point, curr_point: Point) -> str | None:
        """Direction label (the side `curr_point` ended up on) if the step
        prev->curr actually crosses this line segment, else None."""
        if prev_point == curr_point:
            return None
        if not _segments_intersect(self.p1, self.p2, prev_point, curr_point):
            return None
        return self.side_of(curr_point)


@dataclass
class CountResult:
    total: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    by_direction: dict[str, int] = field(default_factory=dict)


class VehicleCounter:
    """Counts unique tracked vehicles crossing one or more CountingLines."""

    def __init__(self, lines: list[CountingLine]):
        self.lines = lines
        self._counted: dict[str, set[int]] = {line.name: set() for line in lines}
        self.results: dict[str, CountResult] = {line.name: CountResult() for line in lines}

    def update(self, tracks) -> list[tuple[str, int, str, str]]:
        """tracks: iterable of Track objects (src.tracking) with >= 2
        trajectory points. Returns new crossing events this call, as
        (line_name, track_id, class_name, direction_label) tuples.
        """
        events: list[tuple[str, int, str, str]] = []
        for track in tracks:
            if len(track.trajectory) < 2:
                continue
            prev_point, curr_point = track.trajectory[-2], track.trajectory[-1]
            for line in self.lines:
                if track.track_id in self._counted[line.name]:
                    continue
                direction = line.crossed_by(prev_point, curr_point)
                if direction is None:
                    continue

                self._counted[line.name].add(track.track_id)
                result = self.results[line.name]
                result.total += 1
                result.by_class[track.class_name] = result.by_class.get(track.class_name, 0) + 1
                result.by_direction[direction] = result.by_direction.get(direction, 0) + 1
                events.append((line.name, track.track_id, track.class_name, direction))
        return events

    def reset(self) -> None:
        for name in self._counted:
            self._counted[name] = set()
            self.results[name] = CountResult()
