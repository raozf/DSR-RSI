"""Ground an explicit color order; preserve the named bottom block in place."""
import re
from ..control import groups, feature, surface, upright, PolicyFailure


def color_order(instruction):
    match=re.search(r'from bottom to top in the order of\s+(blue|green|red|yellow|orange|cyan),\s*(blue|green|red|yellow|orange|cyan),\s*and\s+(blue|green|red|yellow|orange|cyan)\b',instruction,re.I)
    if match is None or len(set(match.groups()))!=3:
        raise PolicyFailure('unsupported_color_order')
    return [x.lower() for x in match.groups()]


def policy(robot,memory):
    order=color_order(robot.instruction)
    scene=yield from robot.observe()
    blocks=groups(scene,'block',3)
    by_color={feature(x,'color'):x['id'] for x in blocks}
    if len(by_color)!=3 or set(order)!=set(by_color):
        raise PolicyFailure('color_identity_ambiguous')
    base=robot.object(by_color[order[0]])
    base_position=list(base['position'])
    support_id=base['id']
    for color in order[1:]:
        yield from robot.observe()
        support=robot.object(support_id);upright(support)
        bottom=robot.object(base['id'])
        if sum((bottom['position'][i]-base_position[i])**2 for i in range(3))**.5>.008:
            raise PolicyFailure('named_base_displaced')
        item=robot.object(by_color[color])
        destination=[*support['position'][:2],surface(support)+item['size'][2]/2]
        yield from robot.transfer(item['id'],destination,support['quaternion'],tolerance=.010)
        support_id=item['id']
    return (yield from robot.home_arms())
