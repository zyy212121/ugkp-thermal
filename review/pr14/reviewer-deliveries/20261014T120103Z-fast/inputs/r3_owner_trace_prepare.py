from r3_runner import *
import difflib
formal=ROOT/'work/fixtures/chemistry-trial-formal/chemistry_trial.cu';text=formal.read_text();new=text
new=new.replace('static int begin(DeviceState* s){++begins;return SharedGasTrialPolicy::begin(s);}',r'''static int begin(DeviceState* s){++begins;const int rc=SharedGasTrialPolicy::begin(s);std::printf("R3TRIAL event=begin attempt=%d rc=%d\n",begins,rc);return rc;}''')
new=new.replace('return rc;\n    }\n    static int applySources',r'''std::printf("R3TRIAL event=validate attempt=%d dt=%a rc=%d stepLimits=%d retry=%d terminal=%d\n",begins,currentDuration,rc,stepLimits,int(s->gasTrial.retryableFailure),int(s->gasTrial.terminalContinuation));
        return rc;
    }
    static int applySources''')
new=new.replace('currentDuration=dt;sourceDurations.push_back(dt);if(!native)++terminalStages;',r'''currentDuration=dt;sourceDurations.push_back(dt);if(!native)++terminalStages;std::printf("R3TRIAL event=source attempt=%d dt=%a native=%d\n",begins,dt,int(native));''')
new=new.replace('if(!rc){++commits;acceptedDurations.push_back(currentDuration);}return rc;',r'''if(!rc){++commits;acceptedDurations.push_back(currentDuration);}std::printf("R3TRIAL event=commit attempt=%d dt=%a rc=%d commits=%d\n",begins,currentDuration,rc,commits);return rc;''')
new=new.replace('static void rollback(DeviceState* s){++rollbacks;SharedGasTrialPolicy::rollback(s);}',r'''static void rollback(DeviceState* s){++rollbacks;SharedGasTrialPolicy::rollback(s);std::printf("R3TRIAL event=rollback attempt=%d dt=%a rollbacks=%d retry=%d terminal=%d\n",begins,currentDuration,rollbacks,int(s->gasTrial.retryableFailure),int(s->gasTrial.terminalContinuation));}''')
new=new.replace('chemistryTrialTest::failedCellTests();chemistryTrialTest::cleanup();','chemistryTrialTest::cleanup();')
assert new.count('R3TRIAL')==5
source=D/'inputs/chemistry-owner-chronology-S304.cu';source.write_text(new);patch=D/'patches/S304-readonly-owner-chronology.patch';patch.write_text(''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='a/formal-generated-chemistry_trial.cu',tofile='b/readonly-owner-chronology.cu')))
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS304只读ownerchronology缺口：原fixture已通过的计数和accepted分段不记录被拒绝attempt的顺序。新独立IO观察只运行原4owner/direct正控制，打印begin/source(dt,native)/validate(stepLimits+decision)/rollback/commit的真实顺序，不改变原assertion/原Co/control或任何GPU调用；16失败矩阵复用正式+S302不在这里重复。观察I/O不计时。保存newsource/patch/bin哈希，completeformalsemanticoutputs按过滤只读行后逐字段核对。compile900/run300。\n')
command=json.loads((ROOT/'work/fixtures/chemistry-trial-formal/command.json').read_text());binary=ROOT/'work/fixtures/chemistry-owner-chronology-S304';command[-3]=str(source);command[-1]=str(binary)
(D/'inputs/chemistry-owner-chronology-S304-command.json').write_text(json.dumps(command,indent=2)+'\n')
t=run('BUILD-CHEMISTRY-OWNER-CHRONOLOGY-S304',cmd(*command),[10,12,14,26],900,'PureIOownersequencewitness,actualnativeformalfixturealgorithm.16failurematrix reusedwithoutrepetition.',inputs=[source,patch],kind='BUILD',variant='DIAGNOSTIC_S304');assert t['exit_code']==0
