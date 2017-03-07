node {
    try {
       stage 'git'
       checkout([$class: 'GitSCM', branches: [[name: '*/master']], doGenerateSubmoduleConfigurations: false, extensions: [[$class: 'SubmoduleOption', disableSubmodules: false, recursiveSubmodules: true, reference: '', trackingSubmodules: false]], submoduleCfg: [], userRemoteConfigs: [[url: 'https://github.com/wichtounet/etl-gpu-blas.git']]])

       stage 'pre-analysis'
       sh 'cppcheck --xml-version=2 --enable=all --std=c++11 include/*.hpp include/egblas/*.hpp test/include/*.hpp test/src/*.cpp src/*.cu 2> cppcheck_report.xml'
       sh 'sloccount --duplicates --wide --details include/ test src > sloccount.sc'
       sh 'cccc include/egblas/*.hpp include/*.hpp test/*.cpp src/*.cu || true'

       env.PATH="${env.PATH}:/opt/cuda/bin/"
       env.CXX="nvcc"
       env.LD="nvcc"

       stage 'build'
       sh 'make clean'
       sh 'make -j3 release'

       stage 'test'
       sh './scripts/test_runner.sh'

       stage 'sonar'
       sh '/opt/sonar-runner/bin/sonar-runner'

       currentBuild.result = 'SUCCESS'
    } catch (any) {
       currentBuild.result = 'FAILURE'
       throw any
   } finally {
       step([$class: 'Mailer',
           notifyEveryUnstableBuild: true,
           recipients: "baptiste.wicht@gmail.com",
           sendToIndividuals: true])
   }
}
