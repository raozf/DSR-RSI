"""Audited native instruction grammars; unknown text never becomes a guess."""
import re
from .geometry import SkillError


def pour_order(instruction):
    """Parse the public three-bottle template; colors still need RGB grounding."""
    if not isinstance(instruction, str):
        raise SkillError('instruction must be text')
    bottle = r'(red|turquoise|violet)'
    bowl = r'(black|white|brown)'
    grammar = (rf'Pour the liquid from the first {bottle} bottle into the first {bowl} bowl, '
               rf'from the second {bottle} bottle into the second {bowl} bowl, '
               rf'and from the third {bottle} bottle into the third {bowl} bowl\. '
               r'Then reset the robot arm\.')
    match = re.fullmatch(grammar, ' '.join(instruction.split()), flags=re.IGNORECASE)
    if match is None:
        raise SkillError('instruction outside audited pour grammar')
    values = [v.lower() for v in match.groups()]
    sources, targets = values[::2], values[1::2]
    if len(set(sources)) != 3 or len(set(targets)) != 3:
        raise SkillError('ambiguous repeated colors require another grounding contract')
    return [dict(order=i, bottle_color=a, bowl_color=b) for i, (a, b) in enumerate(zip(sources, targets))]


def color_pairing(objects, targets):
    """Match distinct observed colors; source and destination IDs are independent.

    Does not classify RGB or calibrate CLIP scores. Repeated/unknown colors
    require another observation strategy, rather than arbitrary tie breaking.
    """
    def validate(items):
        if not isinstance(items, dict) or not items or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in items.items()):
            raise SkillError('nonempty visual ID to color mapping required')
        if len(set(items.values())) != len(items):
            raise SkillError('repeated colors are ambiguous')
    validate(objects)
    validate(targets)
    if set(objects.values()) != set(targets.values()):
        raise SkillError('source and target colors differ')
    reverse = {color: identity for identity, color in targets.items()}
    return {identity: reverse[color] for identity, color in objects.items()}
