import dropbox
# Replace this with your access token
ACCESS_TOKEN = 'sl.u.AFYbfqrbcVJBpMBVoGc4LR3iaEW0--Q1Q8teXtt0tMD5wg_EeqHKu0Y0b83ixvNPflgt5ULfJVxgvTzR7vw3yfhVfJuP-cfFzfCr85xiyKjTOwuhVi6_gGBY3bP0wNRtsON0j-S4YT80jne4kMP0rQxL2HNEIAd1vdu03D-HFXQdT4wCb50vuOd_Z1eqlvbuM1yzO4SlzCGI3cG3mD1Qw4tuXkgO237ISgbQXW1gvSArZKni8r7jVEmf_bneSwque2yHX56wscFbAG7InLnt3IVW5iQomFR0XTqgvPlkdB4zPmU-U5GSymx2FNNtbJ8UVksJUnZlRkuOd1vbhyQX1S_m8liSuYdp-G1rVXzm86BNMCqBT0liSW92b75QxyvEeIppEa6eJCtu2y9weDln1sov5_wHPoen79IkdLvlNLEt9SliVwTxHcr01_7oxvTXNjX52P2nhp3fUpXzSC3WcRPAOMaiHkFaTjZ4yAG0cVdhwXSJxApJ7waDIiIGkWP4i7FXTqeXYLiCowtOmcPEZyjDPKq4dupq1OuFb-Iw6BuiJGp0JLycKkEFsHaoorIKcg7RTNELvFwVBe8ldOSROUqSkJULzfVobRvCCnxFcMZIIAyKy0zn5rk48B_SH02TDrkyY2tohEgD_glJIcmMDo5qKiTi-ETIfWnRLKvzFE7bgt4BPXLvN7sm6Ga6yogQ_esPU7B1n0TQLMCT2r0wyF_KTwQlx5gCQNxlBhc8VOnOCXG0OAFnIT4iMpB82vmDQABQRc3kYnt4wc82-Ogvg3FuF4hGVvN4QpTC9L2cPAK9CUW5vPgv9-LD_vodoNJE_g2uT0nkandLX8W5m7ShoF7u32UZQsPWODniNq9A81wWaebR_rT_CiNI6gkuPrmMnink_EAkR-bYKklpHQYfaFVFizMY3Em98qgRJmRozwov4Z_lZI7Z8xTNQydwNwbETPAfQ2LL947jyhzax7g6UWiy-Q3JE0B2Fv2sP-qYboBtajoOhjuu0v310ByNtj9_7AXGf_bdcGqXgMwNf6nYMk9P3lGbzTNDdP13s19Y03ieRot0n6FJFylXlsuW2qflvZMAO4mv-gyRgl5LptwPgrGU9gIXicSR_L4eUo8rVdg3dfIFzhw0M-4nk4byLizsoDLM0meb1WRKnm8vsqGvPKUz7x5jmMVAzgWEkEEHzjVtTn49vZwE0_-AXN8OFCJsQ3sZJV3PSRoNIzI1vz2DuZruFIu692ZYmW9bHzDzwXDYdqDgDGTMRwc3Ytxy8kHGMWXK69jTUOedhLDJyZWqNaru'

# Initialize Dropbox client
dbx = dropbox.Dropbox(ACCESS_TOKEN)

def get_shared_link(file_path):
    try:
        # Check if a shared link already exists
        result = dbx.sharing_list_shared_links(path=file_path)
        if result.links:
            # If a link exists, retrieve and modify it
            existing_link = result.links[0].url
            modified_link = existing_link.replace('dl=0', 'dl=1')
            return modified_link
        else:
            # If no link exists, create a new one
            link_metadata = dbx.sharing_create_shared_link_with_settings(file_path)
            new_link = str(link_metadata.url).replace('dl=0', 'dl=1')
            return new_link
    except dropbox.exceptions.ApiError as e:
        print("Error handling shared link:", e)
        return None

with open('file_names_only_jpg.txt') as file:
    with open('url.txt', 'w') as u:
        i = 1
        for line in file:
            filename = line.strip()
            image_path = f'test/{filename}'
            image_url = get_shared_link(f'/{filename}')
            u.write(image_url + '\n')