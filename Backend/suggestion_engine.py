# suggestion_engine.py
# ====================
# Generates actionable improvement suggestions based on the evaluation
# results. The suggestions are rule-based at first and can later be
# augmented with a learned model.

from typing import Dict, List, Any


class SuggestionEngine:
    """Builds human-readable improvement suggestions."""

    def __init__(self):
        self.rules = []

    def build_suggestions(self, evaluation: Dict[str, Any]) -> List[str]:
        """Create a list of improvement suggestions."""
        # Simple implementation forwarding to ImprovementRecommender logic
        recommender = ImprovementRecommender()
        metrics = evaluation.get("metrics", {})
        rooms = evaluation.get("rooms", [])
        room_types = [r.get("type", "Living room") for r in rooms]
        
        # Build adjacency map
        adjacency = {}
        for r in rooms:
            adjacency[r["id"]] = r.get("neighbors", [])

        return recommender.generate(
            room_types=room_types,
            adjacency=adjacency,
            space_score=metrics.get("space_utilization", {}).get("space_score", 100),
            adjacency_score=metrics.get("adjacency", {}).get("adjacency_score", 100),
            accessibility_score=metrics.get("accessibility", {}).get("accessibility_score", 100)
        )


class ImprovementRecommender:
    """Generates actionable suggestions for floor plans."""
    
    def generate(
        self,
        room_types: List[str],
        adjacency: Dict[int, List[int]],
        space_score: float,
        adjacency_score: float,
        accessibility_score: float,
    ) -> List[str]:
        suggestions = []
        
        if not room_types:
            suggestions.append("No rooms were detected in the floor plan. Please ensure the uploaded image is a clear 2D floor plan with dark wall lines and distinct, readable room areas.")
            return suggestions
            
        # 1. Space Utilization Suggestions
        if space_score < 70:
            suggestions.append("Consider optimizing space utilization by reducing corridor areas or re-allocating unused corners.")
        
        # Check if we have too much or too little circulation space based on the score
        if space_score < 85:
            has_bedroom = "Bedroom" in room_types
            has_living = "Living room" in room_types
            if has_bedroom and has_living:
                suggestions.append("Ensure rooms are balanced in size relative to each other; avoid making one room disproportionately large.")
        
        # 2. Adjacency Suggestions
        if adjacency_score < 80:
            # Check for specific bad adjacencies
            has_bath_kitchen = False
            has_bath_dining = False
            has_bed_kitchen = False
            
            for src, neighbors in adjacency.items():
                for tgt in neighbors:
                    if src >= len(room_types) or tgt >= len(room_types):
                        continue
                    type_src = room_types[src]
                    type_tgt = room_types[tgt]
                    
                    if {type_src, type_tgt} == {"Bathroom", "Kitchen"}:
                        has_bath_kitchen = True
                    elif {type_src, type_tgt} == {"Bathroom", "Dining room"}:
                        has_bath_dining = True
                    elif {type_src, type_tgt} == {"Bedroom", "Kitchen"}:
                        has_bed_kitchen = True
            
            if has_bath_kitchen:
                suggestions.append("A Bathroom is adjacent to the Kitchen. Consider separating them with a buffer zone or changing door orientations for better hygiene.")
            if has_bath_dining:
                suggestions.append("A Bathroom is adjacent to the Dining room. This can reduce comfort; consider adding a hallway buffer or separating them.")
            if has_bed_kitchen:
                suggestions.append("A Bedroom is directly adjacent to the Kitchen. Noise and cooking odors may disturb sleep; consider adding acoustic insulation or relocating.")

            # Check for missing essential adjacencies
            if "Kitchen" in room_types and "Dining room" in room_types:
                k_indices = [i for i, t in enumerate(room_types) if t == "Kitchen"]
                d_indices = [i for i, t in enumerate(room_types) if t == "Dining room"]
                adjacent = False
                for k in k_indices:
                    for d in d_indices:
                        if d in adjacency.get(k, []):
                            adjacent = True
                if not adjacent:
                    suggestions.append("The Kitchen and Dining room are not adjacent. Connecting them or placing them closer will greatly improve meal serving flow.")

            if "Bedroom" in room_types and "Bathroom" in room_types:
                b_indices = [i for i, t in enumerate(room_types) if t == "Bedroom"]
                bath_indices = [i for i, t in enumerate(room_types) if t == "Bathroom"]
                adjacent = False
                for b in b_indices:
                    for bath in bath_indices:
                        if bath in adjacency.get(b, []):
                            adjacent = True
                if not adjacent:
                    suggestions.append("None of the Bedrooms are adjacent to a Bathroom. Consider relocating a bathroom closer to the sleeping areas for convenience.")

        # 3. Accessibility / Flow Suggestions
        if accessibility_score < 70:
            suggestions.append("Improve floor plan flow by aligning door placements to reduce A* path distances between key rooms.")
            
            # Check for disconnected rooms
            disconnected_rooms = []
            for r_id in range(len(room_types)):
                if not adjacency.get(r_id, []):
                    disconnected_rooms.append(room_types[r_id])
            
            if disconnected_rooms:
                room_names = ", ".join(set(disconnected_rooms))
                suggestions.append(f"We detected isolated rooms (type: {room_names}) with no doors/openings to other spaces. Please add door openings to connect them.")

        if not suggestions:
            suggestions.append("The floor plan layout is well-balanced. Make sure doors open fully without blocking paths.")
            suggestions.append("Verify electrical outlet and lighting placements relative to window positions.")
            
        return suggestions
