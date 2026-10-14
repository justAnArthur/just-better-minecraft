//! Temporary procedural terrain: a local 2D gradient noise with fractal
//! octaves, and a [`Generator`] that turns it into 32³ cubes.
//!
//! This noise is deliberate throwaway scaffolding: task 2.1 replaces it with
//! the declarative density graph. It stays small and self-contained.
//!
//! **Determinism rule:** heights are a pure function of `(seed, x, z)` built
//! only from IEEE-754 `+`, `-`, `*` and `floor` on `f64`, plus integer hashing
//! to pick gradients from a fixed table. Every one of those operations is
//! bit-exact on every platform, so the same seed gives byte-identical terrain
//! everywhere. No `sin`/`cos`/`exp`/`powf` anywhere in this file.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CUBE_SIZE, CubePos, LocalPos};
use jbm_core::cube::Cube;

/// Sea-adjacent level: columns at or below it get sand instead of grass and
/// dirt. There is no water yet (later tasks); this only changes the surface
/// blocks.
const SAND_LEVEL: i64 = 61;

/// Base terrain height before any noise.
const BASE_HEIGHT: f64 = 64.0;

/// Documented height bounds: `BASE_HEIGHT ± 28`, so `[36, 92]`. Every column
/// of the world stays inside this range by construction (the noise deviation
/// is clamped before rounding).
pub const MIN_HEIGHT: i64 = 36;
pub const MAX_HEIGHT: i64 = 92;

/// Amplitude of the rolling-hill noise, scaled so the observed |noise| of
/// about 0.7 reaches the full ±28 deviation.
const HILL_AMP: f64 = 40.0;
/// Hard clamp on the deviation, so heights provably stay within
/// [`MIN_HEIGHT`, `MAX_HEIGHT`] even at the noise's theoretical bound.
const MAX_DEV: f64 = 28.0;
/// Wavelength scale of the large rolling hills (5 fractal octaves).
const HILL_FREQ: f64 = 0.006;
/// Wavelength scale of the low-frequency "hilliness" mask.
const MASK_FREQ: f64 = 0.0009;
/// Octaves of the fractal hill noise.
const OCTAVES: u32 = 5;

/// Fixed table of 256 unit 2D gradients. Lattice points hash into this
/// table; the values are frozen literals so no trig runs at generation time.
#[rustfmt::skip]
static GRADIENTS: [[f64; 2]; 256] = [
    [-0.14085477301109783, -0.9900302686887872],
    [0.4548644075137788, 0.8905607058348909],
    [0.11401442317056107, -0.9934790945506021],
    [0.35259488197622163, -0.9357760678731715],
    [-0.8408130539551082, -0.5413256028479387],
    [-0.48584589676736484, 0.8740444866219997],
    [-0.6273329815419746, 0.7787511350037678],
    [-0.9864350519545125, -0.16415202793599007],
    [0.2892087310698701, -0.9572660601279853],
    [0.915629209124781, -0.4020238194405005],
    [-0.920514608383861, 0.3907081465133622],
    [-0.37122751621966654, 0.9285419383104767],
    [-0.9363206062427931, 0.3511462976098828],
    [-0.12738833799345037, 0.9918529182007111],
    [0.9299103773018058, 0.3677862017346671],
    [0.5387640222668167, -0.8424567219215958],
    [-0.27834513206717937, -0.9604811229037791],
    [0.9878823570488348, -0.15520453805104625],
    [-0.9997925713629252, -0.02036698916163407],
    [-0.7064759528230342, -0.7077370472730574],
    [-0.7005672438366559, -0.7135863906095122],
    [-0.39433157164540505, 0.9189682320970974],
    [-0.08539469094141242, 0.9963472019125766],
    [-0.8822368894248273, 0.47080576774080113],
    [0.3042613343047276, 0.9525885997885482],
    [0.892708639053155, 0.45063431489386585],
    [-0.5125354005604841, 0.8586660952735378],
    [0.766781365612743, 0.6419083558803834],
    [0.8172823399790765, 0.5762374308896683],
    [0.6057593490753991, -0.7956479189991945],
    [0.9547640168201603, 0.2973645442641618],
    [-0.16849049977530453, 0.9857032776071448],
    [0.9023584946075893, -0.4309862494436745],
    [0.7983831579495712, -0.6021497596964315],
    [0.9830628346794427, -0.18326882733301567],
    [0.8446697064810874, 0.5352878542925794],
    [0.9576946995692789, -0.28778614007090186],
    [0.6229583081102148, 0.7822550392017035],
    [0.3520578990683426, -0.9359782239473228],
    [0.9957754989513055, -0.09182132480137016],
    [-0.13320807758786846, -0.9910880929893893],
    [0.4193520066657856, -0.9078237133416261],
    [-0.6492959164088189, 0.7605358722209178],
    [-0.7057813221650041, 0.7084297603030372],
    [-0.361069607216184, -0.932538867149649],
    [0.6665901695007564, 0.745424406579871],
    [0.2856651027226822, 0.9583295096606592],
    [0.9467116140804726, -0.32208247354543595],
    [-0.992879606263464, -0.11912215354881968],
    [0.9995201025404754, 0.030976840017625358],
    [0.999154940559219, -0.041102369227374376],
    [-0.11804854910986752, 0.9930078247692992],
    [0.8518396653058604, 0.5238026198976096],
    [-0.9692627213729186, -0.24602800035110628],
    [0.15089948103252218, -0.9885491118928363],
    [-0.9982967626927117, 0.058340154244324254],
    [0.816710384336825, 0.5770477866835603],
    [-0.6306945327658415, 0.7760311890248206],
    [-0.0789312098811206, 0.9968800650563249],
    [0.7590592414605922, 0.6510215572108733],
    [-0.8881097876129282, -0.45963137963602907],
    [0.9563969863788762, -0.292069862268266],
    [0.9940235317291346, 0.10916601288284826],
    [0.9980552794385124, 0.062335055826662406],
    [-0.03613846315361012, -0.9993467924003635],
    [0.22170765632242848, 0.9751131806759746],
    [-0.4024194667779186, 0.9154553909165513],
    [0.14533678419738963, 0.9893822411783837],
    [0.9986527778739196, -0.051890550630184856],
    [-0.20627919133233158, 0.9784931758695506],
    [0.22559431889720713, 0.9742213317728704],
    [0.8235675419784988, 0.5672182153276583],
    [-0.14073074657497422, -0.9900479064007208],
    [-0.8567993094608887, -0.51565002017584],
    [0.05831177031133788, 0.9982984210360937],
    [-0.22510857773367773, -0.9743336842328304],
    [-0.657100867547639, 0.7538026597645701],
    [-0.9806899641696348, -0.19556889879773925],
    [-0.06280730330541236, 0.9980256723408982],
    [0.5870447216347331, 0.8095545039098966],
    [0.20964806987627152, -0.9777769105461398],
    [0.19844854490569178, 0.9801113074670722],
    [0.9895397410754554, 0.14426053109676534],
    [-0.9597880291457428, 0.2807257364555856],
    [-0.851069520542801, -0.5250530175163711],
    [-0.9845478601963247, 0.17511570741323676],
    [-0.7615999331856925, -0.6480474841950616],
    [-0.5060908046155588, -0.8624802012125127],
    [-0.30223586905137606, -0.9532331716105769],
    [0.9987376018310012, 0.05023149100773773],
    [0.7036640738797807, -0.7105328079201625],
    [0.999524899796707, -0.030821659371012874],
    [0.2576022415094226, -0.966251046658849],
    [-0.09194136435632966, 0.9957644227026273],
    [0.5245533655786092, -0.8513775699771248],
    [0.32898956657692296, -0.9443335560507888],
    [0.8870324765426735, 0.4617070343394947],
    [0.22223088909296473, -0.9749940676398756],
    [-0.9987851190603385, 0.04927764141703012],
    [-0.6853319849833951, 0.7282307809744927],
    [-0.8480780263393087, -0.5298713629178526],
    [0.9973171950812547, -0.07320117755377024],
    [-0.8664424938201057, 0.4992768820031587],
    [-0.2214940644769278, 0.9751617196144907],
    [-0.9975911699588912, 0.06936755451975121],
    [-0.7974050933207427, 0.6034443778395301],
    [0.09842219703208142, 0.9951447488337454],
    [0.9783723918842745, -0.20685130601145252],
    [-0.3698969419527158, 0.9290727917305668],
    [-0.7756398891668531, 0.6311756984653575],
    [0.7858808173934205, 0.6183779918893049],
    [0.16303769468534807, 0.9866198407247277],
    [-0.8688329503738633, -0.4951053467138564],
    [0.48176538179918443, -0.8763002435808666],
    [0.46776031038131055, 0.88385535696288],
    [0.12968489053738264, -0.9915552577472962],
    [-0.7591949006630194, 0.6508633518698591],
    [0.9997783021416083, -0.021055796514094647],
    [0.5899784800038528, -0.8074189700102069],
    [-0.4363553593468028, 0.8997744163785291],
    [-0.16876800817438498, 0.9856558016959321],
    [0.09014844784010613, 0.9959283394662588],
    [-0.9164963461723926, 0.40004305699843606],
    [0.8959971982148547, 0.4440597040839559],
    [-0.9464997416864817, -0.3227045692075391],
    [-0.7558658125629071, 0.6547265638406741],
    [0.9481811223398074, 0.31773032470701157],
    [0.21029486761930602, -0.9776380049143847],
    [0.4718106624389568, 0.8816998915781451],
    [-0.8770305831141276, 0.48043454942634306],
    [0.7982301644341806, -0.6023525583803732],
    [-0.3179881477952041, 0.9480946882362412],
    [0.17622428190811457, 0.9843500406186659],
    [-0.4390324507613718, 0.8984712055366403],
    [0.5791895157343917, -0.8151929249345585],
    [0.8039558969560575, 0.5946889235134458],
    [0.04406207303762439, 0.9990287952404711],
    [-0.9713994998389496, 0.23745107224992396],
    [0.3859515836377535, 0.9225190377913673],
    [0.9007719601549522, -0.4342923851492279],
    [-0.866800154823205, -0.49865568441407326],
    [0.32263183569690207, 0.9465245367104052],
    [0.7584226584126575, 0.651763048359047],
    [-0.868659062267558, -0.49541036882613476],
    [-0.7442298318388899, -0.6679236164420732],
    [0.527492556263191, -0.8495596524593928],
    [-0.4902993870678359, -0.8715540780932096],
    [-0.9743176641088974, 0.22517790612176342],
    [-0.8313956468372551, -0.5556809142125202],
    [-0.05170100359022847, -0.9986626088062791],
    [0.8688372961989255, -0.4950977203883498],
    [0.8247081516631534, 0.5655585421336551],
    [-0.5585113561876512, 0.8294968746230637],
    [-0.9207953966668275, -0.3900459427774884],
    [0.4625478022121654, 0.8865943439187369],
    [-0.4366724718636206, 0.89962056018997],
    [-0.6815830437723975, -0.7317407699738714],
    [0.30372630004203144, 0.9527593267256835],
    [-0.7193009946694399, -0.6946985526597522],
    [0.747459030307042, 0.664307909039518],
    [0.5498544135326255, 0.8352605126059129],
    [-0.4245128780843742, 0.9054218996360323],
    [-0.9657390054959311, -0.2595152659550685],
    [0.8090268552704529, 0.5877716796947619],
    [-0.31344081010254154, 0.9496077393125345],
    [-0.024556714638584994, -0.9996984384133842],
    [-0.49254884738034227, -0.8702847999042017],
    [0.053763113951414504, -0.9985537179232008],
    [-0.9402517808729012, -0.3404799385651639],
    [0.9657037480838588, 0.2596464344773232],
    [-0.9316235043688698, 0.3634248837207855],
    [-0.9958039382636399, -0.0915123846188309],
    [-0.8550260116457699, 0.5185851129844818],
    [0.9980481427617635, -0.06244921720722154],
    [-0.6599690410750559, -0.7512927956678882],
    [0.42653489273904266, 0.904471108038335],
    [-0.706350999078752, -0.707861756348264],
    [-0.9989313169078071, -0.0462193044174653],
    [-0.8192883405962571, 0.5733817358122173],
    [-0.7441177118612899, -0.6680485243560668],
    [-0.9452683554869994, 0.3262939412782648],
    [-0.34035261776506603, 0.9402978759842366],
    [-0.1387373239872719, 0.9903292154293191],
    [0.47188056621449415, 0.8816624814684406],
    [-0.9895378160806196, -0.14427373477663172],
    [-0.9992304383161854, -0.03922411429776284],
    [0.7958380690295985, 0.605509510976698],
    [0.9906551881836368, -0.13639024204408098],
    [-0.8517293790522924, -0.5239819318060466],
    [-0.7293217627426166, -0.6841708605238916],
    [-0.5649269389198451, 0.8251409295887906],
    [0.8819793943356908, -0.4712879671360686],
    [-0.8168339451596047, 0.5768728681737388],
    [-0.48996061298787796, 0.8717445713742891],
    [0.916703398103724, -0.39956836699754567],
    [-0.4553232983959474, -0.8903261727804227],
    [0.9983716862292792, -0.05704363361239947],
    [0.9596887595017275, -0.28106491222853747],
    [0.9996999194491116, 0.024496347757158025],
    [0.387070802830076, -0.9220499951718889],
    [0.25123767459784396, 0.9679254262920609],
    [0.4742342423626085, -0.8803987070474165],
    [-0.1952498090393205, -0.9807535429811656],
    [-0.5061313600907581, -0.8624564025692426],
    [-0.5444500908501847, -0.838793239465618],
    [0.8185677830473785, -0.574409944688373],
    [-0.8644663933677948, 0.5026906153268399],
    [-0.9852501536601409, -0.17112023466752455],
    [0.9937621209495997, -0.11152061229097128],
    [0.8946912454469332, -0.4466850963717229],
    [0.9271425595438103, -0.37470878596925405],
    [0.9847928882369072, 0.17373245890739697],
    [-0.5175300928500248, -0.8556650062931431],
    [-0.9428955911227501, 0.3330884330643735],
    [0.6283982361789137, 0.7778918027381637],
    [-0.9919412828485652, -0.12669842690713537],
    [0.5038259150521455, 0.8638052137616838],
    [0.7055112322993945, 0.7086987378988266],
    [-0.391558017765297, -0.920153421296531],
    [-0.8722156573984159, -0.4891215053430479],
    [0.7311339241083527, 0.6822339664791849],
    [0.7323616599074634, -0.6809158531695271],
    [0.7542592854803123, 0.6565766750858035],
    [-0.9692051248224647, -0.24625479897429528],
    [0.4188067439027643, 0.9080753885342144],
    [0.4310519710588376, 0.9023271015802923],
    [-0.9653562538468737, 0.2609354386792454],
    [0.9210330430223518, 0.38948444598082055],
    [0.879890815277824, -0.4751759181500324],
    [-0.06123839741339856, -0.9981231680921141],
    [-0.04847152591969908, -0.9988245647635103],
    [-0.7421281891446105, 0.6702579733781174],
    [0.9390498747385315, -0.343780937158459],
    [0.32500994939151473, -0.9457105967453918],
    [-0.9994941262266914, 0.03180395633160487],
    [0.8500735314903266, 0.526664021041465],
    [0.6063613556164682, -0.7951892267975963],
    [0.9317917425790011, 0.3629933173786931],
    [-0.4671518120663561, -0.8841771228001322],
    [-0.6729925488126799, 0.7396492609626625],
    [-0.16658281284046705, 0.9860274673994421],
    [-0.48334185851349676, 0.8754316922574364],
    [-0.9072441341752855, -0.42060442342483356],
    [-0.9644460314726396, 0.2642798750882786],
    [0.9776689311929675, 0.2101510432522301],
    [-0.35330750751806506, 0.9355072448310447],
    [-0.4511296970254365, -0.8924584004096425],
    [0.9544941349215759, 0.29822968732222577],
    [-0.7482521923611976, -0.6634143928395445],
    [-0.5087417611045643, 0.8609191718786534],
    [-0.11813428826979054, 0.9929976283632252],
    [0.7917106882107808, 0.6108962155495905],
    [-0.9988128752241244, 0.04871180849155159],
    [0.6139476117057235, -0.7893467742892462],
    [0.7986526093279858, -0.6017923309694131],
    [-0.4266694642714734, 0.9044076339009385],
];

/// Hash a lattice corner and the seed to a gradient index. Integer-only, so
/// it is exact on every platform.
fn gradient_index(seed: u64, x: i64, z: i64) -> usize {
    let mut h = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (z as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 33;
    (h & 0xFF) as usize
}

/// Quintic fade, the classic Perlin smootherstep `6t³ - 15t² + 10t`.
/// Polynomial arithmetic only.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// One sample of 2D gradient noise at `(x, z)`, roughly within
/// `±√2/2 ≈ ±0.707`. A pure function of `(seed, x, z)`.
fn noise2(seed: u64, x: f64, z: f64) -> f64 {
    let x0 = x.floor();
    let z0 = z.floor();
    let fx = x - x0;
    let fz = z - z0;
    let xi = x0 as i64;
    let zi = z0 as i64;
    let g00 = GRADIENTS[gradient_index(seed, xi, zi)];
    let g10 = GRADIENTS[gradient_index(seed, xi + 1, zi)];
    let g01 = GRADIENTS[gradient_index(seed, xi, zi + 1)];
    let g11 = GRADIENTS[gradient_index(seed, xi + 1, zi + 1)];
    let d00 = g00[0] * fx + g00[1] * fz;
    let d10 = g10[0] * (fx - 1.0) + g10[1] * fz;
    let d01 = g01[0] * fx + g01[1] * (fz - 1.0);
    let d11 = g11[0] * (fx - 1.0) + g11[1] * (fz - 1.0);
    let u = fade(fx);
    let w = fade(fz);
    lerp(lerp(d00, d10, u), lerp(d01, d11, u), w)
}

/// Fractal sum of [`noise2`] over [`OCTAVES`] octaves, each half the
/// amplitude and twice the frequency of the last, normalized to roughly
/// `[-0.71, 0.71]`. Doubling a float's exponent is exact, so the frequency
/// ladder adds no rounding.
fn fbm(seed: u64, x: f64, z: f64) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut norm = 0.0;
    for _ in 0..OCTAVES {
        sum += amp * noise2(seed, x * freq, z * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

/// Clamps `t` into `[0, 1]`. `clamp` is a pure comparison — bit-exact.
#[inline]
fn clamp01(t: f64) -> f64 {
    t.clamp(0.0, 1.0)
}

/// Terrain height at column `(x, z)`: `h = 64 + clamp(40 · hilliness ·
/// rolling, ±28)`, rounded with `floor(v + 0.5)` — exact `+` and `floor`
/// arithmetic. `hilliness` is a very low frequency mask that flattens some
/// regions and amplifies others; `rolling` is the 5-octave fractal.
fn height(seed: u64, x: i64, z: i64) -> i64 {
    let xf = x as f64;
    let zf = z as f64;
    let mask = noise2(seed ^ 0x0D1B_54A3_2D16_8ED5, xf * MASK_FREQ, zf * MASK_FREQ);
    let hilliness = clamp01(0.5 + mask * 1.4);
    let rolling = fbm(seed, xf * HILL_FREQ, zf * HILL_FREQ);
    let dev = (HILL_AMP * hilliness * rolling).clamp(-MAX_DEV, MAX_DEV);
    (BASE_HEIGHT + (dev + 0.5).floor()) as i64
}

/// The block at world height `y` in a column whose surface sits at `h`.
#[inline]
fn column_block(y: i64, h: i64, blocks: &Blocks) -> BlockStateId {
    if y > h {
        BlockStateId::AIR
    } else if h <= SAND_LEVEL {
        // Beach and seabed columns: sand takes the grass and dirt layers.
        if y >= h - 3 {
            blocks.sand
        } else {
            blocks.stone
        }
    } else if y == h {
        blocks.grass_block
    } else if y >= h - 3 {
        blocks.dirt
    } else {
        blocks.stone
    }
}

/// How many `(cx, cz)` heightmaps the generator may cache. Bounded, so memory
/// stays flat whatever order requests arrive in.
const HEIGHT_CACHE_CAPACITY: usize = 64;

/// The 32² surface heights of one cube column, indexed `z * 32 + x` to match
/// the cube-local column of the same `(x, z)`.
type Heightmap = Arc<[i64; (CUBE_SIZE * CUBE_SIZE) as usize]>;

/// Generates terrain cubes. Output is a pure function of `(seed, pos)`,
/// whatever order cubes are requested in and however often; the heightmap
/// cache only ever stores results of that same pure function, so it cannot
/// change any answer.
pub struct Generator {
    seed: u64,
    blocks: Blocks,
    height_cache: Mutex<HashMap<(i32, i32), Heightmap>>,
}

impl Generator {
    pub fn new(seed: u64, blocks: Blocks) -> Self {
        Self {
            seed,
            blocks,
            height_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Surface height (block y of the grass/sand layer) at world column
    /// `(x, z)`. Always within [`MIN_HEIGHT`, `MAX_HEIGHT`].
    pub fn surface_height(&self, x: i64, z: i64) -> i64 {
        height(self.seed, x, z)
    }

    /// The spawn position over column `(0, 0)`: centered on the block, two
    /// above the surface.
    pub fn spawn(&self) -> [f64; 3] {
        [0.5, (self.surface_height(0, 0) + 2) as f64, 0.5]
    }

    /// The heights of the 32² columns covered by cube column `(cx, cz)`,
    /// computed at most once thanks to the bounded cache. Consecutive cubes
    /// of one column (streaming works bottom-up-ish per column) hit the
    /// cache.
    fn heightmap(&self, cx: i32, cz: i32) -> Heightmap {
        if let Some(map) = self.height_cache.lock().unwrap().get(&(cx, cz)) {
            return Arc::clone(map);
        }
        let mut map: Box<[i64; 1024]> = Box::new([0; 1024]);
        let base_x = i64::from(cx) * i64::from(CUBE_SIZE);
        let base_z = i64::from(cz) * i64::from(CUBE_SIZE);
        for z in 0..CUBE_SIZE {
            for x in 0..CUBE_SIZE {
                map[(z * CUBE_SIZE + x) as usize] =
                    height(self.seed, base_x + i64::from(x), base_z + i64::from(z));
            }
        }
        let map: Heightmap = Arc::from(map);
        let mut cache = self.height_cache.lock().unwrap();
        if cache.len() >= HEIGHT_CACHE_CAPACITY {
            // Stay bounded: drop the whole (small) working set rather than
            // growing without limit; streaming asks by column, so it refills.
            cache.clear();
        }
        cache.insert((cx, cz), Arc::clone(&map));
        map
    }

    /// The terrain cube at `pos`. Fast paths: a cube whose bottom is above
    /// the column-max height of its 32² area is uniform air, one whose top is
    /// below `min height - 4` is uniform stone — no per-block work at all.
    /// Mixed cubes are built in a single [`Cube::from_fn`] pass, so there is
    /// no per-block `set` palette churn.
    pub fn generate(&self, pos: CubePos) -> Cube {
        let heights = self.heightmap(pos.x, pos.z);
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        for &h in heights.iter() {
            if h < min {
                min = h;
            }
            if h > max {
                max = h;
            }
        }
        let bottom = i64::from(pos.y) * i64::from(CUBE_SIZE);
        let top = bottom + i64::from(CUBE_SIZE) - 1;
        if bottom > max {
            return Cube::default();
        }
        if top < min - 4 {
            return Cube::filled(self.blocks.stone);
        }
        let blocks = self.blocks;
        Cube::from_fn(move |local: LocalPos| {
            let h = heights[usize::from(local.z) * CUBE_SIZE as usize + usize::from(local.x)];
            column_block(bottom + i64::from(local.y), h, &blocks)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Generator, MAX_HEIGHT, MIN_HEIGHT};
    use jbm_core::block::builtin::Blocks;
    use jbm_core::block::{BlockRegistry, BlockStateId};
    use jbm_core::coords::{CubePos, LocalPos};
    use jbm_core::cube::Cube;
    use std::collections::HashMap;

    fn generator(seed: u64) -> Generator {
        let mut registry = BlockRegistry::new();
        let blocks = Blocks::register(&mut registry);
        Generator::new(seed, blocks)
    }

    fn state_ids() -> Blocks {
        let mut registry = BlockRegistry::new();
        Blocks::register(&mut registry)
    }

    #[test]
    fn generating_the_same_cube_twice_gives_equal_cubes() {
        let terrain = generator(42);
        for pos in [
            CubePos { x: 0, y: 2, z: 0 },
            CubePos { x: -3, y: -1, z: 7 },
            CubePos {
                x: 100,
                y: 0,
                z: -100,
            },
        ] {
            assert_eq!(terrain.generate(pos), terrain.generate(pos));
        }
    }

    #[test]
    fn generation_is_order_independent() {
        let terrain = generator(7);
        let mut positions = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    positions.push(CubePos { x, y, z });
                }
            }
        }
        let forward: HashMap<_, _> = positions
            .iter()
            .map(|&p| (p, terrain.generate(p)))
            .collect();
        let backward: HashMap<_, _> = positions
            .iter()
            .rev()
            .map(|&p| (p, terrain.generate(p)))
            .collect();
        // A third order: the first half reversed, then the rest forward.
        let interleaved: HashMap<_, _> = positions
            .iter()
            .take(14)
            .rev()
            .chain(positions.iter().skip(14))
            .map(|&p| (p, terrain.generate(p)))
            .collect();
        assert_eq!(forward.len(), positions.len());
        assert_eq!(backward.len(), positions.len());
        assert_eq!(interleaved.len(), positions.len());
        assert_eq!(forward, backward);
        assert_eq!(forward, interleaved);
    }

    #[test]
    fn different_seeds_give_different_terrain() {
        let a = generator(1);
        let b = generator(2);
        let differing = (-64..64)
            .filter(|&x| a.surface_height(x, x) != b.surface_height(x, x))
            .count();
        assert!(
            differing > 100,
            "seeds 1 and 2 look identical ({differing} differing columns)"
        );
    }

    /// The block at world `(x, y, z)`, through the cube that contains it.
    fn block_at(terrain: &Generator, x: i64, y: i64, z: i64) -> BlockStateId {
        let cube = terrain.generate(CubePos {
            x: (x >> 5) as i32,
            y: (y >> 5) as i32,
            z: (z >> 5) as i32,
        });
        cube.get(LocalPos::new((x & 31) as u8, (y & 31) as u8, (z & 31) as u8).unwrap())
    }

    #[test]
    fn column_layering_is_correct_at_a_sample_column() {
        let terrain = generator(42);
        let blocks = state_ids();
        // Find one grass and one sand column within a bounded window.
        let mut grass = None;
        let mut sand = None;
        'search: for r in 0..4096i64 {
            for &(dx, dz) in &[(r, 0), (-r, 3), (r / 2, -r), (-r / 4, r)] {
                let h = terrain.surface_height(dx, dz);
                if h > 62 && grass.is_none() {
                    grass = Some((dx, dz, h));
                }
                if h <= 61 && sand.is_none() {
                    sand = Some((dx, dz, h));
                }
                if grass.is_some() && sand.is_some() {
                    break 'search;
                }
            }
        }
        let (gx, gz, gh) = grass.expect("a grass column within 4096 blocks of origin");
        let (sx, sz, sh) = sand.expect("a sand column within 4096 blocks of origin");

        // Grass column: air above, grass at h, dirt h-3..h-1, stone below.
        assert_eq!(block_at(&terrain, gx, gh + 1, gz), BlockStateId::AIR);
        assert_eq!(block_at(&terrain, gx, gh, gz), blocks.grass_block);
        assert_eq!(block_at(&terrain, gx, gh - 1, gz), blocks.dirt);
        assert_eq!(block_at(&terrain, gx, gh - 3, gz), blocks.dirt);
        assert_eq!(block_at(&terrain, gx, gh - 4, gz), blocks.stone);
        assert_eq!(block_at(&terrain, gx, gh - 40, gz), blocks.stone);
        // Sand column: sand replaces the grass and dirt layers.
        assert_eq!(block_at(&terrain, sx, sh, sz), blocks.sand);
        assert_eq!(block_at(&terrain, sx, sh - 3, sz), blocks.sand);
        assert_eq!(block_at(&terrain, sx, sh - 4, sz), blocks.stone);
    }

    #[test]
    fn far_above_is_uniform_air_and_far_below_uniform_stone() {
        let terrain = generator(42);
        let air = terrain.generate(CubePos {
            x: 3,
            y: 100,
            z: -2,
        });
        assert_eq!(air, Cube::default());
        let stone = terrain.generate(CubePos {
            x: 3,
            y: -100,
            z: -2,
        });
        assert_eq!(stone, Cube::filled(state_ids().stone));
    }

    #[test]
    fn no_nan_or_extreme_values_at_a_billion() {
        let terrain = generator(42);
        for &(x, z) in &[
            (1_000_000_000i64, 1_000_000_000i64),
            (-1_000_000_000, 1_000_000_000),
            (1_000_000_000, -1_000_000_000),
            (-1_000_000_000, -1_000_000_000),
        ] {
            let h = terrain.surface_height(x, z);
            assert!(
                (MIN_HEIGHT..=MAX_HEIGHT).contains(&h),
                "h = {h} at ({x}, {z})"
            );
            // Generating a whole cube at the far coordinate must not blow
            // up either; heights inside it stay in range.
            let _ = terrain.generate(CubePos {
                x: (x >> 5) as i32,
                y: 2,
                z: (z >> 5) as i32,
            });
        }
    }

    #[test]
    fn heights_stay_in_the_documented_range_over_a_1024_square() {
        let terrain = generator(42);
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        // Every second block of a 1024×1024 area centred on the origin.
        for x in (-512..512).step_by(2) {
            for z in (-512..512).step_by(2) {
                let h = terrain.surface_height(x, z);
                assert!(
                    (MIN_HEIGHT..=MAX_HEIGHT).contains(&h),
                    "h = {h} at ({x}, {z})"
                );
                min = min.min(h);
                max = max.max(h);
            }
        }
        // The range must not be degenerate: hills actually happen.
        assert!(max - min >= 10, "suspiciously flat: [{min}, {max}]");
    }

    #[test]
    fn spawn_sits_two_blocks_over_the_surface() {
        let terrain = generator(42);
        let [x, y, z] = terrain.spawn();
        assert_eq!((x, z), (0.5, 0.5));
        assert_eq!(y, terrain.surface_height(0, 0) as f64 + 2.0);
    }
}
