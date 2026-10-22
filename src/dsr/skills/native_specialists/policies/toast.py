"""Two inserted slices, two protected shelf slices and observed latch."""
import numpy as np
from ..control import groups,feature,PolicyFailure,rotation
from .common import align
from ..resumption import remember_placement,forget_placement

PERCEPTION_PROFILES = ('independent_parts',)


def slot_floor(slot):
    # The visible top opening is above the insertion support. Adding the
    # checker's 3--6.5 cm interval to that opening produces a wrong target.
    if slot.get('reference') not in ('observed_insertion_floor','registered_public_insertion_floor'):
        raise PolicyFailure('toaster_insertion_floor_unresolved')
    point=np.asarray(slot['point'],float)
    if point.shape!=(3,) or not np.isfinite(point).all():
        raise PolicyFailure('toaster_insertion_floor_invalid')
    return point


def check_bread(robot,all_ids,protected,shelf):
    for identity in all_ids:
        item=robot.object(identity)
        normal=rotation(item['quaternion']).apply(feature(item,'face_normal_local'))
        if abs(normal[2])>np.sin(np.deg2rad(25)):raise PolicyFailure('bread_not_upright')
    for identity in protected:
        p=np.array(robot.object(identity)['position']);r=rotation(shelf['quaternion']);local=r.inv().apply(p-shelf['position'])
        if np.any(abs(local[:2])>np.array(shelf['size'][:2])/2+.004):raise PolicyFailure('protected_bread_off_shelf')


def insert_slices(robot,memory):
    scene=yield from robot.observe(required={
        'bread':dict(count=4,features=['face_long_local','face_normal_local','grasp']),
        'toaster':dict(count=1,features=['slots']), 'shelf':dict(count=1)})
    breads=groups(scene,'bread',4);toaster=groups(scene,'toaster',1)[0];shelf=groups(scene,'shelf',1)[0]
    slots=feature(toaster,'slots')
    if len(slots)!=2:raise PolicyFailure('two_slots_required')
    floors=[slot_floor(slot) for slot in slots]
    # Keep two slices already on the shelf; choose the other two by distance.
    def on_shelf(b):
        p=rotation(shelf['quaternion']).inv().apply(np.array(b['position'])-shelf['position'])
        return np.all(abs(p[:2])<=np.array(shelf['size'][:2])/2+.004)
    shelf_breads=[b for b in breads if on_shelf(b)]
    if len(shelf_breads)<2:raise PolicyFailure('two_shelf_slices_not_observed')
    kept=sorted(shelf_breads,key=lambda b:np.linalg.norm(np.array(b['position'])-shelf['position']))[:2]
    chosen=[b for b in breads if b not in kept]
    protected=[b['id'] for b in breads if b not in chosen]
    for identity in protected:robot.protect(identity,.008)
    for bread,slot,floor in zip(chosen,slots,floors,strict=True):
        target=floor+[0,0,.045]
        slot_frame=rotation(slot['quaternion'])
        q=align(feature(bread,'face_long_local'),feature(bread,'face_normal_local'),
                slot_frame.apply([1,0,0]),slot_frame.apply([0,0,1]))
        yield from robot.transfer(bread['id'],target,q,tolerance=.012)
    yield from robot.observe()
    context=dict(bread_ids=[b['id'] for b in breads], protected=protected,
                inserted=[b['id'] for b in chosen], toaster=toaster['id'], shelf=shelf['id'])
    check_slots(robot,context)
    for identity in context['bread_ids']:robot.protect(identity,.008)
    return context


def check_slots(robot,context):
    toaster=robot.object(context['toaster']);slots=feature(toaster,'slots')
    if len(slots)!=2:raise PolicyFailure('two_slots_required')
    for identity,slot in zip(context['inserted'],slots):
        delta=np.array(robot.object(identity)['position'])-slot_floor(slot)
        if np.linalg.norm(delta[:2])>.018 or not .03<=delta[2]<=.065:
            raise PolicyFailure('inserted_slice_not_in_slot',identity)
    check_bread(robot,context['bread_ids'],context['protected'],robot.object(context['shelf']))
    groups(robot.scene,'lever',1)


def finish_lever(robot,memory,context):
    check_slots(robot,context)
    lever=groups(robot.scene,'lever',1)[0];before=np.array(feature(lever,'contact'));bottom=np.array(feature(lever,'bottom'))
    if np.linalg.norm(before-bottom)>.006:yield from robot.press_latch(lever['id'])
    # Releasing the pressing arm is insufficient: require off-contact dwell.
    for _ in range(3):
        yield from robot.wait(3);yield from robot.observe()
        lever_now=robot.object(lever['id'])
        if np.linalg.norm(np.array(feature(lever_now,'contact'))-feature(lever_now,'bottom'))>.007:raise PolicyFailure('lever_rebounded_after_release')
        check_slots(robot,context)
    return (yield from robot.home_arms())


def policy(robot,memory):
    context=yield from insert_slices(robot,memory)
    return (yield from finish_lever(robot,memory,context))


def resume_insert_slices(robot,memory):
    """Revalidate each inserted slice without replaying a released placement."""
    robot.stage='insert_slices:perception'
    scene=yield from robot.observe(required={
        'bread':dict(count=4,features=['face_long_local','face_normal_local','grasp']),
        'toaster':dict(count=1,features=['slots']),'shelf':dict(count=1)})
    breads=groups(scene,'bread',4);toaster=groups(scene,'toaster',1)[0];shelf=groups(scene,'shelf',1)[0]
    key='toast.insert_slices'
    if key not in robot.progress:
        def on_shelf(b):
            p=rotation(shelf['quaternion']).inv().apply(np.array(b['position'])-shelf['position'])
            return np.all(abs(p[:2])<=np.array(shelf['size'][:2])/2+.004)
        kept=sorted([b for b in breads if on_shelf(b)],key=lambda b:np.linalg.norm(np.array(b['position'])-shelf['position']))[:2]
        if len(kept)!=2:raise PolicyFailure('two_shelf_slices_not_observed')
        protected=[b['id'] for b in kept]
        robot.progress[key]=dict(bread_ids=[b['id'] for b in breads],protected=protected,
            inserted=[b['id'] for b in breads if b['id'] not in protected],toaster=toaster['id'],shelf=shelf['id'])
        for identity in protected:robot.protect(identity,.008)
    context=robot.progress[key]
    if set(context['bread_ids'])!={b['id'] for b in breads} or context['toaster']!=toaster['id'] or context['shelf']!=shelf['id']:
        raise PolicyFailure('bread_or_support_identity_changed')
    for index,identity in enumerate(context['inserted']):
        yield from robot.observe()
        slots=feature(robot.object(context['toaster']),'slots')
        if len(slots)!=2:raise PolicyFailure('two_slots_required')
        slot=slots[index];floor=slot_floor(slot);bread=robot.object(identity);target=floor+[0,0,.045]
        slot_frame=rotation(slot['quaternion'])
        q=align(feature(bread,'face_long_local'),feature(bread,'face_normal_local'),
                slot_frame.apply([1,0,0]),slot_frame.apply([0,0,1]))
        if identity not in robot.protected:
            if np.linalg.norm(np.array(bread['position'])-target)<.012 and robot.orientation_error(bread,q)<np.deg2rad(15):
                yield from robot.stable(identity,target,.012,q)
            else:
                remember_placement(robot,identity,target,q,.012,context['toaster'],['slots',index,'point'])
                yield from robot.transfer(identity,target,q,tolerance=.012)
            robot.protect(identity,.008)
            forget_placement(robot,identity)
    yield from robot.observe();check_slots(robot,context)
    return context


STRATEGY_STAGES = (
    dict(name='insert_slices',description='Insert two slices while retaining and checking two shelf slices.',
         run=resume_insert_slices,check=check_slots,protected=lambda c:c['bread_ids'],reserve=180,
         output='toast.inserted/v1',resume='Retain slice/slot identities and protected releases; revalidate before skipping. Empty ownership required.'),
    dict(name='finish_lever',description='Press the observed latch, verify off-contact dwell and return home.',
         run=finish_lever,check_input=check_slots,input='toast.inserted/v1',output='toast.controller_finished/v1',
         terminal=True,resume='Reobserve lever before pressing; retain slice receipt. Withdraw fist contact before retry.'),
)
