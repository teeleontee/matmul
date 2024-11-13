// Creates a test that verifies that matrix multiplication in mode works
//
// * `$name` - test name
// * `$mode` - mode to test, can be basic, easy, medium, hard
// * `$n` - row of first matrix
// * `$m` - col of first matrix
// * `$k` - col of second matrix
// * `$slice1` - raw data of first matrix
// * `$slice2` - raw data of second matrix
// * `$exp` - expected raw data of multiplication result
macro_rules! create_test {
    ($name: ident, $mode: expr, $eq: expr, $n: expr, $m: expr, $k: expr, $slice1: expr, $slice2: expr, $exp: expr) => {
        #[test]
        fn $name() {
            let mut multiplier = $crate::multiplier::implementation($mode).unwrap();

            let m1 = $crate::Matrix::create($n, $m, $slice1).unwrap();
            let m2 = $crate::Matrix::create($m, $k, $slice2).unwrap();

            let res = multiplier.multiply(&m1, &m2).unwrap();
            let expected = $crate::Matrix::create($n, $k, $exp).unwrap();

            if $eq {
                assert_eq!(res, expected);
            } else {
                assert_ne!(res, expected);
            }
        }
    };
}

use crate::args::Mode;

const BASIC: Mode = Mode::Basic;
const EASY: Mode = Mode::Easy {
    device_type: None,
    index: None,
};
const MEDIUM: Mode = Mode::Medium {
    device_type: None,
    index: None,
};
const HARD: Mode = Mode::Hard {
    device_type: None,
    index: None,
};

const M1_1: &[f32] = &[1.0, 2.0, 3.0, 4.0];
const M2_1: &[f32] = &[4.0, 3.0, 2.0, 1.0];
const ANS_1: &[f32] = &[8.0, 5.0, 20.0, 13.0];
const WRONG_ANS_1: &[f32] = &[1.0, 2.0, 3.0, 4.0];

create_test!(test_basic_success_1, BASIC, true, 2, 2, 2, M1_1, M2_1, ANS_1);
create_test!(test_basic_fail_1, BASIC, false, 2, 2, 2, M1_1, M2_1, WRONG_ANS_1);
create_test!(test_easy_success_1, EASY, true, 2, 2, 2, M1_1, M2_1, ANS_1);
create_test!(test_easy_fail_1, EASY, false, 2, 2, 2, M1_1, M2_1, WRONG_ANS_1);
create_test!(test_medium_success_1, MEDIUM, true, 2, 2, 2, M1_1, M2_1, ANS_1);
create_test!(test_medium_fail_1, MEDIUM, false, 2, 2, 2, M1_1, M2_1, WRONG_ANS_1);
create_test!(test_hard_success_1, HARD, true, 2, 2, 2, M1_1, M2_1, ANS_1);
create_test!(test_hard_fail_1, HARD, false, 2, 2, 2, M1_1, M2_1, WRONG_ANS_1);

const M1_2: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
const M2_2: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
const ANS_2: &[f32] = &[30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0];
const WRONG_ANS_2: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

create_test!(test_basic_success_2, BASIC, true, 3, 3, 3, M1_2, M2_2, ANS_2);
create_test!(test_basic_fail_2, BASIC, false, 3, 3, 3, M1_2, M2_2, WRONG_ANS_2);
create_test!(test_easy_success_2, EASY, true, 3, 3, 3, M1_2, M2_2, ANS_2);
create_test!(test_easy_fail_2, EASY, false, 3, 3, 3, M1_2, M2_2, WRONG_ANS_2);
create_test!(test_medium_success_2, MEDIUM, true, 3, 3, 3, M1_2, M2_2, ANS_2);
create_test!(test_medium_fail_2, MEDIUM, false, 3, 3, 3, M1_2, M2_2, WRONG_ANS_2);
create_test!(test_hard_success_2, HARD, true, 3, 3, 3, M1_2, M2_2, ANS_2);
create_test!(test_hard_fail_2, HARD, false, 3, 3, 3, M1_2, M2_2, WRONG_ANS_2);

const M1_3: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
const M2_3: &[f32] = &[1.0, 2.0, 3.0];
const ANS_3: &[f32] = &[14.0, 32.0, 50.0];
const WRONG_ANS_3: &[f32] = &[1.0, 2.0, 3.0];

create_test!(test_basic_success_3, BASIC, true, 3, 3, 1, M1_3, M2_3, ANS_3);
create_test!(test_basic_fail_3, BASIC, false, 3, 3, 1, M1_3, M2_3, WRONG_ANS_3);
create_test!(test_easy_success_3, EASY, true, 3, 3, 1, M1_3, M2_3, ANS_3);
create_test!(test_easy_fail_3, EASY, false, 3, 3, 1, M1_3, M2_3, WRONG_ANS_3);
create_test!(test_medium_success_3, MEDIUM, true, 3, 3, 1, M1_3, M2_3, ANS_3);
create_test!(test_medium_fail_3, MEDIUM, false, 3, 3, 1, M1_3, M2_3, WRONG_ANS_3);
create_test!(test_hard_success_3, HARD, true, 3, 3, 1, M1_3, M2_3, ANS_3);
create_test!(test_hard_fail_3, HARD, false, 3, 3, 1, M1_3, M2_3, WRONG_ANS_3);

use rand::prelude::*;

struct Case {
    m1: crate::Matrix,
    m2: crate::Matrix,
}

impl Case {
    fn test_case(&self, mode: Mode) {
        // we assum that the basic multiplier is always correct
        let mut basic_multiplier = crate::multiplier::implementation(BASIC).unwrap();
        let mut multiplier = crate::multiplier::implementation(mode).unwrap();

        let actual = multiplier.multiply(&self.m1, &self.m2).unwrap();
        let expected = basic_multiplier.multiply(&self.m1, &self.m2).unwrap();

        assert_eq!(actual, expected);
    }
}

fn generate_case() -> Case {
    let mut rng = rand::thread_rng();
    let n = rng.gen::<usize>() % 100;
    let m = rng.gen::<usize>() % 100;
    let k = rng.gen::<usize>() % 100;

    let mut m1 = crate::Matrix::create_empty(n, m);
    let mut m2 = crate::Matrix::create_empty(m, k);

    m1.iter_mut().for_each(|el| *el = rng.gen::<f32>() % 1000.0);
    m2.iter_mut().for_each(|el| *el = rng.gen::<f32>() % 1000.0);

    Case { m1, m2 }
}

#[test]
fn random_tests() {
    for _ in 0..5 {
        let case = generate_case();
        // test gpu implementations
        case.test_case(EASY);
        case.test_case(MEDIUM);
        case.test_case(HARD);
    }
}
