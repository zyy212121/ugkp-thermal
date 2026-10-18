from test_current_gas_courant import launches

def test_switch_labels_stay_outside_emulated_launch_lambda():
    source = "switch(capacity){case 24:ns::wall<true,24><<<workers,32,0,stream>>>(s);break;default:wall<false><<<workers,32>>>(s);}"
    converted = launches(source)
    assert "case 24:emulateLaunch([&](){ns::wall<true,24>(s);},workers,32);" in converted
    assert "default:emulateLaunch([&](){wall<false>(s);},workers,32);" in converted
