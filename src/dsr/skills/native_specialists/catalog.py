"""Superseding executable candidate inventory; historical designs stay intact."""
SPECS = {
 'pour_by_language':('language_pour','pour-by-language',800),
 'fill_egg_holder':('eggs','general-pickup',700),
 'arrange_largest_number_random':('digits','arrange-largest-number-random',1050),
 'arrange_largest_number':('digits','arrange-largest-number-random',1050),
 'sort_nesting_dolls_by_size_random':('dolls','sort-nesting-dolls-by-size',1050),
 'fold_clothes_random':('fold','general-pickup',500),
 'stack_blocks_random':('stack','general-pickup',550),
 'stack_blocks':('stack','general-pickup',550),
 'pack_objects_into_box_random':('pack','general-pickup',1300),
 'pack_objects_into_box':('pack','general-pickup',1300),
 'hang_mugs_random':('mugs','general-pickup',800),
 'hang_mugs':('mugs','general-pickup',800),
 'make_toast_random':('toast','general-pickup',1400),
 'make_toast':('toast','general-pickup',1400),
 'store_laptop_and_headphones_random':('laptop','general-pickup',800),
 'store_laptop_and_headphones':('laptop','general-pickup',800),
 'sweep_blocks_random':('sweep','general-pickup',1000),
 'sweep_blocks':('sweep','general-pickup',1000),
}

QUERIES = {
 'eggs':[('egg','egg',4),('egg_holder','egg carton base',1),('lid','egg carton lid',1)],
 'stack':[('block','cube block',3)],
 'fold':[('garment','long sleeve shirt',1)],
 'pack':[('box','open cardboard box',1),('car','toy car',1),('brush','electric toothbrush',1),('hammer','hammer',1),('shoe','shoe',1)],
 'mugs':[('mug','mug',3),('rack','cup holder',1)],
 'toast':[('bread','slice of bread',4),('toaster','toaster',1),('shelf','bread shelf',1),('lever','toaster lever',1)],
 'laptop':[('headset','headphones',1),('stand','headphone stand',1),('laptop','laptop',1),('laptop_rack','vertical laptop storage rack',1)],
 'sweep':[('broom','broom',1),('pan','dustpan',1),('target','brick . earbuds . paper ball',None)],
}


def limits(task):
    return dict(max_steps=SPECS[task][2],max_commands=900,home_reserve_steps=65,
        recover_perception=True,recover_motion=True,
        clearance_m=.08,lift_m=.07,grasp_tolerance_m=.020,stability_m=.006,
        contact_offset_local=[.145,0,0],grasp_quaternion=[.5,-.5,.5,.5],
        max_translation_step_m=.018,max_rotation_step_rad=.12)
