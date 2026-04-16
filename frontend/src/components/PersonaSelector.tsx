import { useEffect, useState } from "react";
import type { Persona } from "../types";
import { fetchUsers } from "../lib/api";

interface Props {
  selected: string | null;
  onSelect: (persona: Persona) => void;
}

export function PersonaSelector({ selected, onSelect }: Props) {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchUsers()
      .then(setPersonas)
      .catch(() => setError("Cannot reach API — start the FastAPI server"));
  }, []);

  if (error) return <p className="error">{error}</p>;

  return (
    <div className="persona-selector">
      <label htmlFor="persona">Persona</label>
      <select
        id="persona"
        value={selected ?? ""}
        onChange={(e) => {
          const p = personas.find((p) => p.id === e.target.value);
          if (p) onSelect(p);
        }}
      >
        <option value="" disabled>Select a persona</option>
        {personas.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} ({p.condition})
          </option>
        ))}
      </select>
    </div>
  );
}
