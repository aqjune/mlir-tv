// VERIFY
// ARGS: -max-const-tensor-size=1

#map0 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @transpose() -> tensor<1x1x16x96xf32> {
  %cst = arith.constant dense<"0x9EE8A93C1BE6963CD4F3D73C832620BCC6F7E6BD42FAB23C5E86BCBC385D00BE0C2D01BEDB7CD63C3A11B13A5F8502BE88915C3CEC537A3D5B55AC3BE861A4BD4BB0B83DCFAFFEBD145C563C0617003CC75A7A3E6273C23C8A6278BC36FB173D184E5EBD35DB6E3D11C2A53D2A9ED03BF04D8C3D39F724BD20A8673DED72DCBB6D01C0BD3F88A6BDDA056DBDA032503D92BF223DD98768BD2A31613DF804063E50220A3E2D5B83BCCDB1F63B9105073E74F59CBD40B1B7BD70DE93BD35FDB83D468C94BD8E32C13D0C14833D7716C83DF181E5BD513806BD04EDAF3D5D976DBC2C1412BDBD0DDA3DE62F7B3ED9461DBD6FE935BE60A51ABE490CBD3C0147C8BCC72CE7BD2EF084BB1188ECBD6508833CDAB41D3DEF14C8BD246A0EBC583A2ABD21348FBA39BEA03B1F0A333DB33F36BD9BCAF23DC5BC5CBDB1EF66BE6ACA9EBD684E33BB64FAF1BC7AF6CD3C1F7097BBB43EB53C08048B3BFD2EDC3BBCAA48BC523021BC19B0EEBC3137953CC835063D8F63F83B1208C63C61CD85BCD9B4963CCF0B183C04D493BCD0287FBB7D616439900E6E3D5FB3193C21DBEFBB2B3126BE18021D3EA3C6163C32D28C3CFFDE203D66207E3BA71E0ABBD322A03CBC83CEBC5CF99CBAAC99C1BC1294A6BBEBA189BC5BF89A3DFBD4E5BA075EAE3C224A993C387806BC1441B43A766BB53CD78BA33D051CA73B8B8291BC296052BCDD27873DB835AE3B3F12D9BC4B1896BDAC6B54BD01A3B13BDFBDEFBA6F8C48BD461710BB5B92D83B09CD1FBD672A4DBD7C01A83B271A903C3DE7063C7DC786BC276F5BBBA8290F3EC1C4533B0B9ABDBCD8C9DBBB831A2A3A561AE03CCF9FDD3DCB5FF1BBB63C183B40D858BCC9BB08BD0CFCC73B58E017BD805C673BD7C753BCECCE1C3CD125003C46A6843C387FFBBB24F081BBAF16893CD00A98BC51FA813C5A8B9BBD94A9E13D793ED4BA26CD163C32BB81BB1287763BAF368A3C05E3AE3C6951C4BA9C5497BCADDC40BC977D3A3DA35CA63BD3AD853A0A8F7B3D870477BCDD5EECBC1870493CC128E2BC8E4919BD803E533CB41287BCEA349BBDE80B13BD6CC57BBC9F0F053C1BD4673B4AB2133D2351623CCA6BC13C340C0F3D32998DBC9D330ABD87983CBBD2D7E9BCB2490CBD890671BE519049BC0EB372BDC33ABDBCDC358B3E17F1A83AD451AB3C74619FBBF231973CA8D880BD87207BBBDAA874BC365576BCBE9BEE3C2341A6BB01CEC7BCABB59CBD40A936BCD5D0963C6A137F3CEB028EBD7C17733AAF2CE93CFAA695BCF266BEBB3554EB3BA5150DBD89EFC03B036E053D5F39DCBC66D713B9AD14013CB2018CBC959CB3BB9137383DB68A263C511B793C8335BA3C9B02893D462DB7BD042A69BC89900C3D509712BE3F24CE3CB68E4FBDC585DD3D364FCD3C324A803DFAABC4BC4261203CC386363E7D5B9A3DB0EE223D963DF2BBDDF8BFBC05678C3D3521A6BC7A71653B88D6AABB0C350C3DCC9907BC1F0DA43BF21462BD9E75D33C2CD1083D49712D3DBC7C533D48A77FBD0F89A43C8A634BBC76E4283C2D0D26BD5C680CBC731E193D74F2BC3CA355973CFA18423B650EDB3AEB7CB33CF361D4BBDD80FF3BB713D0BCEE76C6BC696FD13C63CFFB3B1D72773B5D2937BE35B4F23D9008503DA9F0A2BD3E126F3EBEE26DBD6633D3BAFF26A5BD193D233CC20B37BDCB87803E160FBB3CC13D84BE94A4AABD449E8ABE8F2E22BF52C971BE98A4333E08F2DBBE42C1723D8127DF3DE2CABA3EA92B48BE0FA4BDBD3084BBBE8DE16A3E5FC4563E548F8DBEAB7AB43EFBE518BE4D9C1F3ED2E23BBD8A98913DA891B73D7D5FA23D256780BA735601BD32B78D3DE1F586BBC3BE17BBA57A703D33B9D93DC460C1BB8B9691BD2EA03CBD8853253DB7BE0CBC9BA89F3A82F07A3D0EA08DBD64DFF3BD7E9E6D3C1500603DFDB962BD255E2CBB89DD78BC5999113D5493DE3D0976D2BC404001BD202C5FBDFB413EBD13A81C3C6D96793EB359303EB33DA23E87D039BDC16345BDD25B873E4912C7BE899F62BFEF4A56BF3F9F813B22E743BD8D5246BF1242C63C21C1093F5B3EE33D12D91FBF53C509BC259967BBA048923D114D263B5B9E823C2A618E3D04FDBABC7B3498BD106E9BBBBC1DEDBC869363BD205AA6BECDF9E1BC9983A2BDD14363BD6801873EB17CA2BB639F8FBC43BF17BC01D2ACBBC1D70F3D0DF0073C6E0C293C137F833DB5304E3D1E4C213CEA00133B305B66BC225B0F3CEC5AA2BCB96AB2BC7F6C903A881FB03B084B333C6CEE6A3C56B758BB37C304BCF979C7BB09EAB6BAB62E2DBDBA0150BDD2E906BCE37F473ACBA8323C7E6C1AB9EE29AB3BCEBE953C9390FD3A652ECDBBCE83E03BCE0A7BBC8E4580BDCA6BE3BCDD904C3D83F673BCC1CC693D7CAAD6BD3CEFB43CE74C853D678E83BDFD0522BB43B0CCBC5AC935BDDE22233D703E1CBC368638BC5D9CB039C56F14BCB0CB423D3D9C27BBA4B7823C0AEC61BB2F8837BCC9C6473A62708D3B76420E3CA9538A3B805CE4BB98B7AD3BAFADFA3BBD0FF4BC3833363D82BF8DBDEDEF9D3DA40AF23B9946773D4C2CD43C403037BCFAD61FBC7B226ABCF5AC443C1FCA2A3B44465C3D337BE536230033BB99D40CBB96AD473A26889FBD6A0F773D6C0A37BDE787943D1162E13D436FA2BCFD9C663FE3E97B3F833384BEEBBF923D804B9CBF2BC429BE1CF13BBC8ABB813EC04C61BFB7F332BC8691853BFD68D93A16B493BC0C78313D1B250F3CA38B3D3D3E19A93DBF6EEB3D2AD0CE3CDF1A8D3CE4B7EA3CBA193E3C7EDEFCBC5C3E2CBA2E1B6E3D048CF8BB111C38BD56F6843CC1F79FBCAE0ABD3CF5D9CA3B1E724D3C1F17AABC2329023A195CD33B54DDFA3CA2290C3E64EDB03C93579DBDC2E6C3BC56A3F23DF7B2A3BEBE8FDEBED5DBA6BED45D063FA49BA83DEE9208BF98E7EABD0987D4BDC379AEBD635A34BEA85755BC72C19D3B3F3006BFA34A61BE672220BDADD31ABEC2F7163DCB8B5CBD42D6F7BBD4B782BD6EC9F33B25CB713DC3C604BD7F442B3C22A7843C53A7BC3AE1FE3ABDEEDB143DEDABB23DD66B013D273F0DBC65721C3CDC5D0A3C2BFAF2BBEB3B55BB2C27473BC75E10BE90F183BC6C14B4BCEE5182BBFD0B90BB1906BEBC0472983C623E193D2525F6BCF50AA6BA2FC66DBC0925EF3C5BF240B960D78B3C9E45143DAD7F183B0D670F3AA6FD113D7A1A96BB3566153E3DE54FBEA0E1F9BC440D0ABD029D2FBC938442BC402522BD276635BD73FFE3BB4F8964BDC53CAA3D3AAC93BDA358D5BDAC8642BE58FA60BD6B5D583E5BC3E33E057CC23E7EEA863D7ED6E43CFE00D33E6C3EE23DE48393BE147DC63CBED5CA3EDD9A39BB52BF5FBC46F4013C150644BCF8553EBC4BC470BC7C1061BC2CCC5FBDF966A1BDB5657D3CBF2DA53C545DC83D7FEA8B3CFC9415BCEBCAB2BC1D889B3D9F73BE3D5A0E4D3C384A4B3F561D7E3D73582EBE07429B3E25B8453E7758493E2C7C883DD7ED423E351CD3BE22F4D73D4089573FD0BDDB3E0C2E0C3FFC33113E57D80DBF4C89713E891EEEBE164A023FF74ABC3CDC97C03E70A4003ED215A2BD14BAD3BDA75E983EBD1C95BD320B8DBA0A92C73E4EB9143C090022BD771B9DBDE811633C02903F3EE2A900BEAE141FBFE884D9BC3CC3D53EA8C29FBE0B578A3DE3D9913D30DD1BBBA07A02B78635E63C8FC5B13E9E47323E27ECDBBD58F9BD3DE5D5463CC9139B3D07C39EBDE3DEACBC3B930D3DD4BE93BD5ABBB93C98459A3D526D2F3D366CB0BB1C1F1B3D8819EA3E6205133C7217E53D8272853D01F7D5BED510213A065D0139DFCD1D3BEE95343BFB8679BB224E0A3BB2A7113CDBAA843DCF90A5BDA0558FBB12651FBB497635BC77D3063AA0FA78BB8DE223BC9EC5A53A333D3EBD43AE203C9F9D51BDEB8A853DF34595BB4691823D1CFB4E3CD9ABB3BA922B8ABCF6605BBD67E7463D8D1CA63A9CCE643D44BCEAB77282EE3C3644653B9ECD243CBD59333C9254623C280E3D3CD722FDBCFAD585BC4AC198BC45D22A3C6623D63B9BFD093C73C151BC36D1A6BCD3D6153C626A94BBCB5E2B3B14D195BC5F3327BEA9BF51BDE35286BEDBB101BD7008D23B5A1B193E71C8A93D3C546EBDFA2B85BD348EB63DEFB281BDA9BB883C5A6931BE6E14B4BC1250D03DCD2184BBDBE8473CAB179BBC0903A63D78E1BDBD03DFE63A2B1385BD7C0E09BD9D2B5F3C8981BF3C10143BBD3D7D9B3CA0A9CDBB811979BD7B642D3B7380CA3CA263AA3BC9E022BE0DF3923DECE1AEBC2CE20EBE5BACDA3CE73CB53DB590453EA996FB3ED4B1EB3EB4442D3D88CECA3DF07DDB3E43DBF13C7CC33EBE8FAC22BEF2ABB53E2607983C242C0A3BACC3E63C2B7C72BCD79AAEBCB54432BD2026553D60EBA5BD283F8CBD02FD3CBD3D0B12BC904ADD3C7B54B63BF3A91FBD5A51D23D1438113CD6B45F3C641FD03C02D039BC78F300BC132C153DCBF6BDBB895FBF3B849EA93BDA35983CB2C42BBC8355BC3A736B553EEC73553A5B3CA13CB5A1313C83536BBEE3FD01BD118104BD3A38183C82F5B6BCC040F93D1EA44DBC398D233D5BD8DC3BA65BD8BC2E16813C55089F3BBBD78A3A0064853BD2F49CBB8619A73BE6A9583AFF7D28BDCFACD2BC044864BD3801E13C0AF2CB3BE746223D24EB133D924BD93DDDB0BB3DCF0BB63D05E307BD55902CBE7B8C7D3CCC5EE0BC4CD6B83CC563F7BDEF9FA2BC06799DBD482679BD0142653D95A6733DC225213D0EB6783D4113A43D9EC3143D59A4E03DD6AFB33B45AC7ABDCD02883D27749CBCA88799BB558D68BD88EEDB3BF2FB9A3C5527DF3C59435E3B350F7EBDAEE3D13CB9433EBC82AB633C00ED06BB4C88363BC7D388BC5A147E3CCDA2C6BCB2529D3CA34D913CC7A56E3C1793043D6E314D3D8C8EE93C272A9EBD1BE99ABCC1A5983D4D35043CFA28963C439AAD3CBF4D283D35668F3CDB71023C04ABAD3DACEEF53C7B09803B2A9F923CCCADE7BAEA892DBD988E1D3D4DF480BDD0634E3D5DA506BC0027293DDD655D3CA9A9093B0C708BBD831B473D5E64F9BC33F2533C6815E1BB9C01183DC62AA3BCE987F63A8A0D483D8AC0003B9B24C13A8DA3B8BCF02B0BBD05F7CFBC76C0E2BD5F014FBDF5CBC8BCF73719BB1245503CADEBE6BCC7DF613DD3EBA43CEE466C3C21188BBC84129DBDE50A46BC430CBCB9BFDDB6BCCD156FBD72CC2B3C5E0795BC7C271CBD2E064C3D1B173B3D31A402BF4ACF1D3C3BDDC33C1A0FE43B12AA003F836ED8BB536A6F3873F75D3C7A94E53BA4A7F33BE6635ABBDB211EBC4DC000BD9162D93B4C4158BCC14A6D3A444BCBBE95FB46BBF2D50EBC22F1B2BB3BCDCD3E9E058C3CDE919B3CDAF194BDFC6CFF3DCC2F003D454C5D3D5A6AC5BDD7AD8CBCDA8296BB4DC405BDA808CEBC50C692BE4B49D6BD7F17D23D04348D3D33C992BEF3AA8ABC5A816A3D81AE573DE3A949BCEB2077BC2D78263DC16F603CF6EA863E1F6518BE7C75AEBD0C7CABBDD9200BBDEF3B81BC7702F4BD7BAD00BD17D501BC60F546BE8B4F373E5E3830BD24A4373E611FAA3DC3F8913ED730ACBE8760953E776FB03E2781553E47BF41BD8787C5BDFD591BBEF7FE873E445F9CBE3EB28BBBE7088DBB49C49E3D1C83D63DA6AFC1BCFBCEDCBE18D64D3C3C70913D22251DBCD890C1BC7195EF3D99ACE73C8883953C3C23F53D1BC0B43DE9D40F3D5BDB99BD4C0DF73B9A12ADBD465E233EB84641BE106082BCBEA932BE704539BD127EE9BAB0A6363B63E1443E801FC1BDC7B096BAA5DC0BBEE1DB13BC408184BDE16B11BC044555BC16F49CBCAA57B4BC58C03FBBB0322B3DEBB063BC38C2433D64990E3EEC20D33D7AD6533CFA50A23C36F9513C265901BB2645FEBCCE41263BCD06953DB647863D16443E3D5991F73CA22E783D8A8F81BCA7F18CBD9114A8BD4082B63B3E20FDBA2BE95F3DA92BBABC23C43ABC6E40553DAB4F85BDEA28803C8F0781BC94ADAB3CE7D790BCCEA54C3C82DAAB3C2D2D7BBB9C67173C58C7F5BC92C78BBDCB4A96BDA144073BF743A8BB3C5F94BD7EC466BC2464423C5175773AE97E88BDE149C1BB95F2EF3D7EFB90BC57AA953A258AE2BAC6E09DBDEB04793C1088103F570E1DBF41A06ABDA25183BCE8A7BBBDFAAA0EBDA957EE3D705847BDAE5F373D686A94BCF95406BD4E2B933C194CEFBCA95F3F3ECE3F063CDB2D4B3D245B27BC9B5D0EBD91E405BB3F37923C28F104BCD37F223DD435E3BBCC74EA3C34B5F6BB66FF363BB24C65BCD58E4B3D6DB6203D04FB3B3C1A17DD3D7178A6BDC5992D3DF658203DB73FA63DBE3311BD4B28433D9AEE59BD844B013EBDBBE039D212DB3CEE40E33D161B053F8D752B3FF9A3AA3EC401BBBB8B402D3CBE4E013F45CEEBBC9C62513D676F833EA2BCB03E207714BD1578E3BD29DCFABD48E32E3E60EEB53CAB60D83C9ADF49BCC726AF3B58EE753C1C61FABC0F24F0BB73FECEBC4531C2BD9E279CBD18BD6BBC993A813CC180EC3CEF1089BC9EF3DD3CEC67EBBBDCB1083DF220663BBC4CAB3CE35FA8BBD1850D3CF5AA33BDEA5508BC53D328BC2D71823BB4541D3CC2B203B9875054BC89A50BBD46230CBA341C263C7A1A1A3CC28EFBBC2D476EBB87EA9EBB8929CA3B8008F13AA35B69BC5C4DF23C54B0ADBCBB881E3D1BA40D3DBECB0A3D88EA35BBF41D6F3B9D4528BC541A063D5DBDB7BC2735983BB790123EAB2BCA3E0ED750BE94A08D3EBCCE8CBCC87E48BDEF1FCEBE47D80D3EE861273E2672903E0B19A9BEC6794FBEFEE4DF3D9C8A7DBE0888783E1C6D2DBED4F87DBA2C49F8BCE849D63C13E4C53B418E6FBCC6C64A3DD05F2ABC789733BD11CE193D1220AC3973002DBCB80025BC6B50753B29796ABD34364CBC1628ED3A4B5FE6BCD6C277BD2323913D445C2C3DECA1753AA2914F3A673F20BB1784B93C420FA73C477A9ABC7CD9DABD4E4BBF3C4C10AB3DCAD7B73DCA76663DADEA173C7C1B62BD9DE49C3C3470843D3700C43D56D606BE18B1B3BDBF50943D700894BB57ED0BBD472F0E3D36E4883DAFF22BBD324DA7BD2BB56DBD157CF43C3752D8BC9C808DBC3878DEBCD5EA813D59FB8ABD8A6FAE3BF5356FBD1A66DDBC755B053A4908603C49D3143D9B35A4BCC393F6BB99EE5FBD91013E38FD561FBC59E92DBC6F3C143D8DE0583EA4B1A0BDDBD0893E2CFE11BDF3A89CBB2337C1BE7BC959BFE84E66BFD5D69E3E6F9D16BBF8979E3F28D1143EF4F1BF3D20938FBEC3F25C3F8F0BB8BB48ED99BD34278A3D24CDB53C83CA253D8EE9DF3DA1A5C8BCAAFCCBBD2A2E743D2E539ABB5E4E26BDF3E76B3C3B7DAFBBFC2B05BE2B4DE3BC1D0FF1BC4F02DFBBC59D51BDE0C0A03CDA0D41BD5051B03C827B18BDC4AFA13D154B39BD7B6273BD596F29BD6D03063D0478DA3C64AE0C3D7B9C96BC7F8C5DBCA20EBA3C73A7B0BEA4A74E3EB2BE8BBD8D439EBE9C0B513DF212413EAAB6A93E8A32713FF4065B3F1F99C63DC7B04A3E0C244A3F8712843DECB267BEDCCBA9BE993C3F3F13910D3F5CF94CBE7A198D3E66C424BEA53393BD3B8E3ABD8A4FBDBEA1CDB83D87F8963B36ED67BF60F91D3F03A10A3E710A15BECB8446BC21E69E3E38804E3E7D8FBE3B81E393BCA42D18BC7F08E93BC55851BE8AEBE93A57F3353BD09C5A3C0F8B293CA005F83BD4707E3BC7EC783B16CB713BE4E9493B802523BCA05B0EBA50E64A3DCAB0EBBB36B3643DECBB2C3DA460913D0F2AFABD27CBE0BD7413E1BE08CD08BF6A43F23D8595333D3D842B3FC93DEB3D516E543CD9F845BE335CED3E5E5B23BEA147BE3C45E0973D630EB7BE1B36D7BD6040D0BE8CDC903EF7BEA7BE327990BE1766ACBE7EA337BCF955743CC7BA413E54C5B1BE79DCDA3E8536AFBAFECCD2BDB390473D95918D3D437578BD91F69BBC1835F6BDC08E84BD4E39E9BEC430E0BE560166BC9862C8BA96A5233FCED32A3DDBFBD93C9563FBBD9F41E23EB4B83EBBFFB3EB3C61E392BC725E00BC04C19E3C344014BD288B403C98F0243D154104BD6FD54EBB38FFBA3BBA23F63B871746BBEC28213D17D8F43B50DB9D393262EEBCEFF2E7BCDD768F3D010F96BD012E033CD6C492BD432F37BD9015113CE2D3453CDA764E3DA4CFB5BC19418BBC50B35EBD59982B3C2D45A8BBFCBB32BC45DE13BC071A49BD35204D3D0D6149BDC244E93C25431EBDFAD130BDF72D4F3C1A1AF53C619D633D7872253BFA3B9ABD3179A1BCECC2ACBB36A6E5388214B13C58D0423CB224D2BD80D35EBCCA094B3CBA8282BBC2AF9A3CB50A99BC77EAB7BE0E2CC93E68C8923DF608603D67A7A43CD3AB353DE51411BC81C1803D247D71BB79FBC6BDAB7EA6BD6B290B3E5A6424BEFC9DE9BC72F23BBE2B0D07BDE81959BC634AA9BCAACFE23EC7D85DBE4E17D1BB54B005BE1A7953BCA8B207BE421ECEBC7AE3B43A6832B1BB4301943B002080BBDA08C63CD897113C8B6B183C8617D83CB019F73C01B1E73AB5B21CBC74A092BD24C88CBA926EAE3B0BAF863CC17448BD22CE543F46ABD2BD3BE112BE8B0E67BD873AD73C7B9BA3BC544BEE3E69DF81BDCDD15D3DE3242DBEC8DC43BE43BDD9BBA7E3B6BC7F673A3DEE03CCBEE90704BD6EA3933BAC4BD53C58A11F3C527693BB02270BBD868313BC59B6053CD25D15BD7D07DABCD520C6BCDA207DBBCEC39A3C16B8D4BB02A638BBCBBCE43C89EB993C"> : tensor<96x1x1x16xf32>
  %1 = linalg.init_tensor [1, 1, 16, 96] : tensor<1x1x16x96xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : tensor<96x1x1x16xf32>) outs(%1 : tensor<1x1x16x96xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<1x1x16x96xf32>
  return %2 : tensor<1x1x16x96xf32>
}
