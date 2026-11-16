You annotate short maintenance work orders for named entities.

Each word of the record is shown with its index, for example `0:<id> 1:pump 2:leaking`.
Mark every entity as a span `[start, end, type]` over word indices: `start` is the index of the
first word and `end` is the index after the last word, so `[1, 2, "PhysicalObject"]` is the single
word at index 1.

Types (use exactly these names):
- PhysicalObject: equipment, parts, components and materials, e.g. "engine", "air conditioner", "hose", "oil", "tyres".
- State: a condition or fault of an object, e.g. "broken", "unserviceable", "cracked", "fault", "not working", "worn".
- Process: something happening to or in an object, e.g. "leaking", "leak", "overheating", "weeping".
- Activity: maintenance work done or needed, e.g. "replace", "change out", "repair", "inspect", "service".
- Property: a property or feature of an object, e.g. "crack", "noisy", "power".

Rules:
- Spans may be nested: mark a longer name and the shorter name inside it when both are objects,
  e.g. "air filters" and "filters".
- Placeholders such as `<id>`, `<num>`, `<date>` and `<sensitive>` are never entities.
- If there is no entity, answer `{"spans": []}`.
- Answer with one JSON object and nothing else: `{"spans": [[start, end, "Type"], ...]}`.

Examples:

{examples}

Record:
{record}
