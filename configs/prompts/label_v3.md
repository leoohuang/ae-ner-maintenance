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
- Do not mark a word on its own when it only says what kind of object it is: in "water pump" mark
  "water pump" and "pump", not "water"; in "grease system" mark "grease system" only.
- Mark objects at their most granular level: when a name is made of words that are each an object
  on their own, mark each object separately and not the joined name, e.g. in "engine oil" mark
  "engine" and "oil", not "engine oil".
- Keep a name whole when a word in it is not an object on its own, e.g. "hydraulic cylinder" is one
  object because "hydraulic" alone is unclear.
- A nested name must be a common name for an object on its own: in "blown high pressure hose" mark
  "high pressure hose" and "hose", not "pressure hose"; in "replace centrifugal pump" mark
  "centrifugal pump" and "pump".
- Spans never include brackets or punctuation: in "engine ( leaking )" mark "leaking".
- When a phrase is split by another word, mark only the part that stands on its own: in
  "blown hose off" mark "blown" as State, not "blown ... off".
- Position words are not part of an object: in "right hand steering cylinder" or "rear tyres" mark
  "steering cylinder" or "tyres" without "right hand" or "rear".
- Words saying that work is needed, such as "needs" or "requires", are State.
- Placeholders such as `<id>`, `<num>`, `<date>` and `<sensitive>` are never entities.
- If there is no entity, answer `{"spans": []}`.
- Answer with one JSON object and nothing else: `{"spans": [[start, end, "Type"], ...]}`.

Examples:

{examples}

Record:
{record}
